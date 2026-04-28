"""Providers are the classes that actually do the API calls to the different ASR services."""

import asyncio
import io
import json
import traceback
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import aiofiles
import aiohttp
from aiopath import AsyncPath
from pydantic import BaseModel, HttpUrl, SecretStr
from rich.progress import Progress, TaskID

from rtasr.asr.options import (
    AssemblyAIOptions,
    AwsOptions,
    AzureOptions,
    DeepgramOptions,
    GoogleOptions,
    RevAIOptions,
    SpeechmaticsOptions,
    WordcabHostedOptions,
    WordcabOptions,
)
from rtasr.asr.schemas import (
    ASROutput,
    AssemblyAIOutput,
    AssemblyAIUtterance,
    AssemblyAIWord,
    AwsOutput,
    AzureOutput,
    DeepgramAlternative,
    DeepgramChannel,
    DeepgramOutput,
    DeepgramResult,
    DeepgramUtterance,
    DeepgramWords,
    GoogleOutput,
    RevAIElement,
    RevAIMonologue,
    RevAIOutput,
    SpeechmaticsOutput,
    SpeechmaticsResult,
    WordcabHostedOutput,
    WordcabHostedTranscript,
    WordcabOutput,
    WordcabTranscript,
)
from rtasr.concurrency import ConcurrencyHandler, ConcurrencyToken
from rtasr.utils import build_query_string


def _debug_log(debug: bool, provider_name: str, message: str) -> None:
    """Print a tagged debug message when ``debug`` is enabled.

    Centralized so every provider/debug line uses the same prefix and is easy
    to grep in long logs (e.g. ``rtasr ... --debug 2>&1 | grep DEBUG``).
    """
    if debug:
        from rich import print as rich_print

        rich_print(
            rf"[bold magenta]\[DEBUG][/bold magenta] [{provider_name}] {message}"
        )


class GatewayTimeoutError(Exception):
    """Exception raised when the API call times out."""

    def __init__(self, status_code) -> None:
        """Initialize the exception."""
        self.status_code = status_code
        super().__init__(f"Gateway Timeout: Status Code {status_code}")


class ProviderConfig(BaseModel):
    """The base class for all ASR provider configurations."""

    api_url: HttpUrl
    api_key: Union[SecretStr, None]  # Wordcab self-hosted without auth


class ProviderResult(BaseModel):
    """The base class for all ASR provider results."""

    cached: int
    completed: int
    errors: List[str]
    failed: int
    provider_name: str


class TranscriptionStatus(str, Enum):
    """Status of the transcription."""

    CACHED = "CACHED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    IN_PROGRESS = "IN_PROGRESS"


class ASRProvider(ABC):
    """The base class for all ASR providers."""

    def __init__(
        self, api_url: str, api_key: str, concurrency_limit: Union[int, None]
    ) -> None:
        """
        Initialize the ASR provider.

        Args:
            api_url (str):
                The URL of the ASR provider API.
            api_key (str):
                The API key of the ASR provider.
            concurrency_limit (Union[int, None]):
                The maximum number of concurrent API calls. If `None`, there is no
                limit.
        """
        self.config = ProviderConfig(api_url=api_url, api_key=api_key)
        self.concurrency_handler = ConcurrencyHandler(limit=concurrency_limit)

        self.max_retries = 3
        self.provider_name = self.__class__.__name__.lower()

    @property
    @abstractmethod
    def output_schema(self) -> ASROutput:
        """The output schema of the ASR provider."""
        return ASROutput

    async def launch(
        self,
        audio_files: Dict[str, List[Path]],
        output_dir: Path,
        session: aiohttp.ClientSession,
        split_progress: Progress,
        split_progress_task_id: TaskID,
        step_progress: Progress,
        data_range: Union[str, None],
        use_cache: bool,
        debug: bool,
    ) -> ProviderResult:
        """
        Call the API of the ASR provider.

        Args:
            audio_files (Dict[str, List[Path]]):
                The audio files to transcribe with the ASR provider. The keys are the
                names of the splits and the values are the list of audio files to
                transcribe for each split.
            output_dir (Path):
                The output directory where to save the results.
            session (aiohttp.ClientSession):
                The aiohttp session for the API calls.
            split_progress (Progress):
                The progress bar for the split progress. It is used to track the
                progress of the transcription of each split.
            split_progress_task_id (TaskID):
                The task ID of the split progress bar. It is used to update the
                progress of the split progress bar.
            step_progress (Progress):
                The progress bar for the step progress. It is used to track the
                progress of the transcription of all the audio files for a
                specific ASR provider.
            data_range (Union[str, None]):
                The range of the data to transcribe. It is used to restrict the
                transcription to a specific range of the data. If `None`, the
                transcription is done on the full data range.
            use_cache (bool):
                Whether to use the cache or not. If `True`, the ASR provider will
                not transcribe the audio files that are already in the cache.
                The RTTM files will be generated from the cache if not already
                present.
            debug (bool):
                Whether to run in debug mode or not. If `True`, the ASR provider
                will only transcribe the first audio file of each split, no
                matter data range.
                This is useful for debugging or testing the full process.

        Returns:
            ProviderResult:
                The result of the ASR provider. It contains the name of the provider,
                the number of files that were successfully transcribed and the number
                of files that failed to be transcribed.
        """
        _debug_log(
            debug,
            self.provider_name,
            f"launch() called. splits={list(audio_files.keys())}, "
            f"file_counts={ {k: len(v) for k, v in audio_files.items()} }, "
            f"output_dir={output_dir}, use_cache={use_cache}, "
            f"data_range={data_range}, api_url={self.config.api_url}",
        )

        if debug:
            audio_files = {
                split_name: audio_files[split_name][0:1]
                for split_name in audio_files.keys()
            }
            _debug_log(
                debug,
                self.provider_name,
                "Debug-mode slicing applied: keeping only the first audio file "
                f"per split. New file_counts={ {k: len(v) for k, v in audio_files.items()} }",
            )
        elif data_range:
            start, end = data_range.split(":")
            start, end = int(start), int(end)
            audio_files = {
                split_name: audio_files[split_name][start:end]
                for split_name in audio_files.keys()
            }

        task_tracking: Dict[str, Any] = {}

        # ``tasks`` must accumulate across splits: previously this list was
        # reassigned at every iteration, which meant every split's coroutines
        # except the last one were silently dropped (RuntimeWarning:
        # "coroutine 'ASRProvider._launch' was never awaited") and only the
        # last split was actually awaited by ``as_completed`` below.
        tasks: List[Callable] = []
        step_progress_task_ids: Dict[str, Any] = {}
        step_progress_totals: Dict[str, int] = {}

        for split_name, split_audio_files in audio_files.items():
            split_task_count_before = len(tasks)
            _debug_log(
                debug,
                self.provider_name,
                f"Preparing tasks for split='{split_name}' "
                f"({len(split_audio_files)} audio file(s)).",
            )
            for audio_file in split_audio_files:
                task_tracking[audio_file.name] = {
                    "audio_file_name": audio_file.name,
                    "error": None,
                    "split": split_name,
                    "status": TranscriptionStatus.IN_PROGRESS,
                }
                _check_cache_task = await self._check_cache(
                    audio_file=AsyncPath(audio_file),
                    output_dir=AsyncPath(output_dir / split_name),
                )
                (
                    asr_output_exists,
                    rttm_file_exists,
                    dialogue_file_exists,
                ) = _check_cache_task

                _debug_log(
                    debug,
                    self.provider_name,
                    f"[{split_name}] cache check for '{audio_file.name}': "
                    f"asr_output_exists={asr_output_exists}, "
                    f"rttm_file_exists={rttm_file_exists}, "
                    f"dialogue_file_exists={dialogue_file_exists}",
                )

                if use_cache and asr_output_exists:
                    task_tracking[audio_file.name]["asr_output_cache"] = True

                    if not rttm_file_exists or not dialogue_file_exists:
                        tasks.append(
                            self._get_asr_output_from_cache(
                                audio_file=audio_file,
                                output_dir=output_dir / split_name,
                            )
                        )
                        _debug_log(
                            debug,
                            self.provider_name,
                            f"[{split_name}] '{audio_file.name}' -> queued "
                            "_get_asr_output_from_cache (rttm/dialogue missing).",
                        )
                    else:
                        task_tracking[audio_file.name][
                            "status"
                        ] = TranscriptionStatus.CACHED
                        _debug_log(
                            debug,
                            self.provider_name,
                            f"[{split_name}] '{audio_file.name}' -> fully cached, "
                            "skipping API call.",
                        )

                    task_tracking[audio_file.name]["rttm_cache"] = rttm_file_exists
                    task_tracking[audio_file.name][
                        "dialogue_cache"
                    ] = dialogue_file_exists

                else:
                    task_tracking[audio_file.name]["asr_output_cache"] = False
                    task_tracking[audio_file.name]["rttm_cache"] = False
                    task_tracking[audio_file.name]["dialogue_cache"] = False
                    tasks.append(
                        self._launch(
                            audio_file=audio_file,
                            url=self.config.api_url,
                            session=session,
                            debug=debug,
                        )
                    )
                    _debug_log(
                        debug,
                        self.provider_name,
                        f"[{split_name}] '{audio_file.name}' -> queued _launch "
                        "(will hit the API).",
                    )

            split_task_count = len(tasks) - split_task_count_before
            _debug_log(
                debug,
                self.provider_name,
                f"[{split_name}] {split_task_count} task(s) queued for execution"
                f" (running total: {len(tasks)}).",
            )

            step_progress_task_ids[split_name] = step_progress.add_task(
                "",
                action=(
                    f"[bold yellow][ {split_name} ][/bold yellow]"
                    f" [bold green]{self.__class__.__name__}[/bold green]"
                ),
                total=split_task_count,
            )
            step_progress_totals[split_name] = split_task_count

        _debug_log(
            debug,
            self.provider_name,
            f"Awaiting {len(tasks)} task(s) across "
            f"{len(step_progress_task_ids)} split(s) via asyncio.as_completed...",
        )

        try:
            for future in asyncio.as_completed(tasks):
                task_result = await future
                audio_file_name, status, asr_output = task_result
                _debug_log(
                    debug,
                    self.provider_name,
                    f"Task completed: '{audio_file_name}' status={status} "
                    f"output_type={type(asr_output).__name__}",
                )

                if (
                    status == TranscriptionStatus.CACHED
                    or status == TranscriptionStatus.COMPLETED
                ):
                    task_tracking[audio_file_name]["status"] = status
                    _split = task_tracking[audio_file_name]["split"]

                    if not task_tracking[audio_file_name]["rttm_cache"]:
                        rttm_lines = await self.result_to_rttm(asr_output=asr_output)
                        await self._save_rttm_files(
                            audio_file_name=audio_file_name,
                            rttm_lines=rttm_lines,
                            output_dir=output_dir / _split,
                        )

                    if not task_tracking[audio_file_name]["dialogue_cache"]:
                        dialogue_lines = await self.result_to_dialogue(
                            asr_output=asr_output
                        )
                        await self._save_dialogue_files(
                            audio_file_name=audio_file_name,
                            dialogue_lines=dialogue_lines,
                            output_dir=output_dir / _split,
                        )

                    if not task_tracking[audio_file_name]["asr_output_cache"]:
                        await self._save_asr_outputs(
                            audio_file_name=audio_file_name,
                            asr_output=asr_output,
                            output_dir=output_dir / _split,
                        )

                elif status == TranscriptionStatus.FAILED:
                    task_tracking[audio_file_name]["status"] = status
                    if isinstance(asr_output, Exception):
                        task_tracking[audio_file_name]["error"] = str(asr_output)

                _split_for_progress = task_tracking[audio_file_name]["split"]
                _step_id = step_progress_task_ids.get(_split_for_progress)
                if _step_id is not None:
                    step_progress.advance(_step_id)

        except Exception as e:
            print(
                f"[bold red]Problem with {self.__class__.__name__} -> {e}[/bold red]]"
            )
            # If there is an exception, we mark all the tasks still in progress
            # as failed.
            for task in task_tracking.values():
                if task["status"] == TranscriptionStatus.IN_PROGRESS:
                    task_tracking[task["audio_file_name"]][
                        "status"
                    ] = TranscriptionStatus.FAILED

        # Force-complete every per-split progress bar in case some tasks
        # failed/were skipped before reaching ``step_progress.advance`` above.
        for _split_name, _sid in step_progress_task_ids.items():
            step_progress.update(
                _sid, completed=step_progress_totals[_split_name]
            )
        split_progress.advance(split_progress_task_id)

        status_counts = Counter(task["status"] for task in task_tracking.values())

        return ProviderResult(
            provider_name=self.__class__.__name__,
            completed=status_counts[TranscriptionStatus.COMPLETED],
            failed=status_counts[TranscriptionStatus.FAILED],
            cached=status_counts[TranscriptionStatus.CACHED],
            errors=[
                f"{task['audio_file_name']} -> {task['error']}"
                for task in task_tracking.values()
                if task["error"]
            ],
        )

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
        debug: bool = False,
        **kwargs,
    ) -> Tuple[str, TranscriptionStatus, ASROutput]:
        """Run the ASR provider."""
        retries = 0
        concurr_token: Union[ConcurrencyToken, None] = None

        _debug_log(
            debug,
            self.provider_name,
            f"_launch() entering for '{audio_file.name}' (max_retries="
            f"{self.max_retries}). Waiting for concurrency token...",
        )

        while retries < self.max_retries:
            try:
                concurr_token = await self.concurrency_handler.get()
                _debug_log(
                    debug,
                    self.provider_name,
                    f"Got concurrency token for '{audio_file.name}'. "
                    f"Calling get_transcription() (attempt {retries + 1}/"
                    f"{self.max_retries})...",
                )

                results = await self.get_transcription(
                    audio_file=audio_file,
                    url=url,
                    session=session,
                    debug=debug,
                    **kwargs,
                )
                status, asr_output = results
                _debug_log(
                    debug,
                    self.provider_name,
                    f"get_transcription() returned for '{audio_file.name}': "
                    f"status={status}",
                )

                retries = self.max_retries  # To break the while loop

            except (
                aiohttp.client_exceptions.ClientOSError,
                aiohttp.client_exceptions.ServerDisconnectedError,
                GatewayTimeoutError,
            ) as e:
                retries += 1
                _debug_log(
                    debug,
                    self.provider_name,
                    f"Retryable error for '{audio_file.name}' "
                    f"(retry {retries}/{self.max_retries}): {type(e).__name__}: {e}",
                )
                if retries >= self.max_retries:
                    status = TranscriptionStatus.FAILED
                    asr_output = Exception(f"{e}\n{traceback.format_exc()}")
                    break
                else:
                    print(
                        f"Retrying {audio_file.name} for {self.__class__.__name__}..."
                    )
                    await asyncio.sleep(1)

            except Exception as e:
                _debug_log(
                    debug,
                    self.provider_name,
                    f"Non-retryable exception for '{audio_file.name}': "
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                )
                status = TranscriptionStatus.FAILED
                asr_output = Exception(f"{e}\n{traceback.format_exc()}")
                break

            finally:
                if concurr_token is not None:
                    self.concurrency_handler.put(concurr_token)
                    concurr_token = None

        return audio_file.name, status, asr_output

    async def _check_cache(
        self, audio_file: AsyncPath, output_dir: AsyncPath
    ) -> Tuple[bool, bool]:
        """Check the cache for the audio file.

        This method check if the audio file has already been transcribed and is
        in the cache. It will check the asr output file, the RTTM file for the
        DER evaluation and the dialogue file for the WER evaluation.

        Args:
            audio_file (Path):
                The audio file to check.
            output_dir (Path):
                The output directory where the results are saved, i.e. the cache.

        Returns:
            Tuple[bool, bool, bool]:
                A tuple of booleans indicating if the asr output file, the RTTM
                file and the dialogue file are in the cache.
        """
        _file_name = audio_file.name.split(".")[0]
        asr_output_file_path = (
            output_dir / f"{self.provider_name}" / "original" / f"{_file_name}.json"
        )
        rttm_file_path = (
            output_dir / f"{self.provider_name}" / "rttm" / f"{_file_name}.rttm"
        )
        dialogue_file_path = (
            output_dir / f"{self.provider_name}" / "dialogue" / f"{_file_name}.txt"
        )

        asr_output_exists = await asr_output_file_path.exists()
        rttm_exists = await rttm_file_path.exists()
        dialogue_exists = await dialogue_file_path.exists()

        return asr_output_exists, rttm_exists, dialogue_exists

    async def _get_asr_output_from_cache(
        self, audio_file: Path, output_dir: Path
    ) -> Tuple[str, TranscriptionStatus, ASROutput]:
        """Get the asr output from the cache and return it.

        This function return the same output as the `_launch` method but it
        loads the asr output from the cache instead of calling the API.

        Args:
            audio_file (Path):
                The audio file to load.
            output_dir (Path):
                The output directory where the results are saved, i.e. the cache.

        Returns:
            Tuple[str, TranscriptionStatus, ASROutput]:
                The same output as the `_launch` method but with the asr output
                loaded from the cache.
        """
        raw_asr_output = await self._load_cache(audio_file, output_dir)

        asr_output = self.output_schema.from_json(raw_asr_output)

        return audio_file.name, TranscriptionStatus.CACHED, asr_output

    async def _load_cache(self, audio_file: Path, output_dir: Path) -> dict:
        """Load the cache for the audio file.

        This method load the asr output file from the cache.

        Args:
            audio_file (Path):
                The audio file to load.
            output_dir (Path):
                The output directory where the results are saved, i.e. the cache.

        Returns:
            dict:
                The raw asr output loaded from the cache.
        """
        _file_name = audio_file.name.split(".")[0]
        asr_output_file_path = (
            output_dir / f"{self.provider_name}" / "original" / f"{_file_name}.json"
        )

        async with aiofiles.open(asr_output_file_path, mode="r") as f:
            data = await f.read()

        raw_data = json.loads(data)

        return raw_data

    async def _save_asr_outputs(
        self,
        audio_file_name: str,
        asr_output: ASROutput,
        output_dir: Path,
    ) -> None:
        """
        Save the asr outputs to disk.

        Args:
            audio_file_name (str):
                The name of the audio file.
            asr_output (ASROutput):
                The output of the ASR provider to save.
            output_dir (Path):
                The output directory where to save the results.
        """
        _file_name = audio_file_name.split(".")[0]
        asr_output_file_path = AsyncPath(
            output_dir / f"{self.provider_name}" / "original" / f"{_file_name}.json"
        )
        if not await asr_output_file_path.parent.exists():
            await asr_output_file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(asr_output_file_path, mode="w") as f:
            await f.write(
                json.dumps(asr_output.model_dump(), indent=4, ensure_ascii=False)
            )

    async def _save_dialogue_files(
        self,
        audio_file_name: str,
        dialogue_lines: List[str],
        output_dir: Path,
    ) -> None:
        """
        Save the dialogue files to disk.

        Args:
            audio_file_name (str):
                The name of the audio file.
            dialogue_lines (List[str]):
                The dialogue lines to save.
            output_dir (Path):
                The output directory where to save the results.
        """
        _file_name = audio_file_name.split(".")[0]
        dialogue_file_path = AsyncPath(
            output_dir / f"{self.provider_name}" / "dialogue" / f"{_file_name}.txt"
        )
        if not await dialogue_file_path.parent.exists():
            await dialogue_file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(dialogue_file_path, mode="w") as f:
            await f.write("\n".join(dialogue_lines))

    async def _save_rttm_files(
        self,
        audio_file_name: str,
        rttm_lines: List[str],
        output_dir: Path,
    ) -> None:
        """
        Save the RTTM files to disk.

        Args:
            audio_file_name (str):
                The name of the audio file.
            rttm_lines (List[str]):
                The RTTM lines to save.
            output_dir (Path):
                The output directory where to save the results.
        """
        _file_name = audio_file_name.split(".")[0]
        rttm_file_path = AsyncPath(
            output_dir / f"{self.provider_name}" / "rttm" / f"{_file_name}.rttm"
        )
        if not await rttm_file_path.parent.exists():
            await rttm_file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(rttm_file_path, mode="w") as f:
            await f.write("\n".join(rttm_lines))

    @abstractmethod
    async def get_transcription(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
        debug: bool = False,
    ) -> Tuple[TranscriptionStatus, ASROutput]:
        """Make API calls to the ASR provider and return the result."""
        raise NotImplementedError(
            "The ASR provider must implement the `get_transcription` method."
        )

    @abstractmethod
    async def result_to_dialogue(self, asr_output: ASROutput) -> List[str]:
        """Convert the result to dialogue format for WER."""
        raise NotImplementedError(
            "The ASR provider must implement the `result_to_dialogue` method."
        )

    @abstractmethod
    async def result_to_rttm(self, asr_output: ASROutput) -> List[str]:
        """Convert the result to RTTM format for DER."""
        raise NotImplementedError(
            "The ASR provider must implement the `result_to_rttm` method."
        )


class AssemblyAI(ASRProvider):
    """The ASR provider class for AssemblyAI."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = AssemblyAIOptions(**options)

    @property
    def output_schema(self) -> AssemblyAIOutput:
        """The output format of the AssemblyAI ASR provider."""
        return AssemblyAIOutput

    async def get_transcription(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
        debug: bool = False,
    ) -> Tuple[TranscriptionStatus, Union[AssemblyAIOutput, Exception]]:
        """Call the API of the AssemblyAI ASR provider."""
        headers = {
            "Authorization": f"{self.config.api_key.get_secret_value()}",
        }

        async with aiofiles.open(audio_file, mode="rb") as f:
            async with session.post(
                url=f"{url}/upload",
                data=f,
                headers=headers,
            ) as response:
                if response.status == 500:
                    raise GatewayTimeoutError(response.status)
                elif response.status == 401 or response.status == 400:
                    raise Exception(await response.text())
                else:
                    content = (await response.text()).strip()

        upload_url = json.loads(content).get("upload_url")
        payload = {"audio_url": upload_url, **self.options}

        async with session.post(
            url=f"{url}/transcript", json=payload, headers=headers
        ) as response:
            if response.status == 500:
                raise GatewayTimeoutError(response.status)
            elif response.status == 401 or response.status == 400:
                raise Exception(await response.text())
            else:
                content = (await response.text()).strip()

        transcript_id = json.loads(content).get("id")

        while True:
            async with session.get(
                url=f"{url}/transcript/{transcript_id}", headers=headers
            ) as response:
                if response.status == 500:
                    raise GatewayTimeoutError(response.status)
                elif response.status == 401 or response.status == 400:
                    raise Exception(await response.text())
                else:
                    content = (await response.text()).strip()

            body = json.loads(content)
            if body.get("status") == "completed":
                asr_output = AssemblyAIOutput.from_json(body)
                status = TranscriptionStatus.COMPLETED
                break
            elif body.get("status") == "error":
                asr_output = Exception(body.get("error"))
                status = TranscriptionStatus.FAILED
                break
            else:
                await asyncio.sleep(3)

        return status, asr_output

    async def result_to_dialogue(self, asr_output: AssemblyAIOutput) -> List[str]:
        """Convert the result to dialogue format for WER."""
        utterances: List[AssemblyAIUtterance] = asr_output.utterances

        dialogue_lines: List[str] = []
        if not utterances:  # This means there is only one speaker
            words: List[AssemblyAIWord] = asr_output.words
            dialogue_lines.append(" ".join([word.text for word in words]))
        else:
            for utterance in utterances:
                dialogue_lines.append(utterance.text)

        return dialogue_lines

    async def result_to_rttm(self, asr_output: AssemblyAIOutput) -> List[str]:
        """Convert the result to RTTM format for DER."""
        utterances: List[AssemblyAIUtterance] = asr_output.utterances

        rttm_lines: List[str] = []
        if not utterances:  # This means there is only one speaker
            words: List[AssemblyAIWord] = asr_output.words
            rttm_lines.append(f"{words[0].start / 1000} {words[-1].end / 1000} A")
        else:
            for utterance in utterances:
                start_seconds: float = utterance.start / 1000
                end_seconds: float = utterance.end / 1000
                speaker: str = utterance.speaker

                rttm_lines.append(f"{start_seconds} {end_seconds} {speaker}")

        return rttm_lines


class Aws(ASRProvider):
    """The ASR provider class for AWS."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = AwsOptions(**options)

    @property
    def output_schema(self) -> AwsOutput:
        """The output format of the AWS ASR provider."""
        return AwsOutput

    async def get_transcription(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
        debug: bool = False,
    ) -> Tuple[TranscriptionStatus, AwsOutput]:
        """Call the API of the AWS ASR provider."""
        raise NotImplementedError("Aws not implemented.")

    async def result_to_dialogue(self, asr_output: AwsOutput) -> List[str]:
        """Convert the result to dialogue format for WER."""
        pass

    async def result_to_rttm(self, asr_output: AwsOutput) -> List[str]:
        """Convert the result to RTTM format for DER."""
        pass


class Azure(ASRProvider):
    """The ASR provider class for Azure."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = AzureOptions(**options)

    @property
    def output_schema(self) -> AzureOutput:
        """The output format of the Azure ASR provider."""
        return AzureOutput

    async def get_transcription(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
        debug: bool = False,
    ) -> Tuple[TranscriptionStatus, AzureOutput]:
        """Call the API of the Azure ASR provider."""
        raise NotImplementedError("Azure not implemented.")

    async def result_to_dialogue(self, asr_output: AzureOutput) -> List[str]:
        """Convert the result to dialogue format for WER."""
        pass

    async def result_to_rttm(self, asr_output: AzureOutput) -> List[str]:
        """Convert the result to RTTM format for DER."""
        pass


class Deepgram(ASRProvider):
    """The ASR provider class for Deepgram."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        """Initialize the Deepgram ASR provider."""
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = DeepgramOptions(**options)

    @property
    def output_schema(self) -> DeepgramOutput:
        """The output format of the Deepgram ASR provider."""
        return DeepgramOutput

    async def get_transcription(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
        debug: bool = False,
    ) -> Tuple[TranscriptionStatus, DeepgramOutput]:
        """Run the Deepgram ASR provider."""
        headers = {
            "Authorization": f"Token {self.config.api_key.get_secret_value()}",
            "Content-Type": f"audio/{audio_file.suffix[1:]}",
        }

        _url = f"{url}{build_query_string(self.options)}"
        try:
            audio_size = audio_file.stat().st_size
        except OSError:
            audio_size = -1
        _debug_log(
            debug,
            self.provider_name,
            f"POST {_url} | file='{audio_file.name}' "
            f"size={audio_size} bytes | content_type={headers['Content-Type']} | "
            f"options={dict(self.options) if hasattr(self.options, 'items') else self.options}",
        )

        async with aiofiles.open(audio_file, mode="rb") as f:
            _debug_log(
                debug,
                self.provider_name,
                f"Opened audio file '{audio_file.name}'. Sending request "
                "(this is where the run will appear stuck if the server "
                "doesn't respond)...",
            )
            async with session.post(url=_url, data=f, headers=headers) as response:
                _debug_log(
                    debug,
                    self.provider_name,
                    f"Response received for '{audio_file.name}': "
                    f"status={response.status}",
                )
                if response.status == 200:
                    content = (await response.text()).strip()
                elif response.status == 504:
                    raise GatewayTimeoutError(response.status)
                else:
                    raise Exception(await response.text())

        _debug_log(
            debug,
            self.provider_name,
            f"Response body length for '{audio_file.name}': {len(content)} chars",
        )

        if not content:
            status = TranscriptionStatus.FAILED
            asr_output = None
        else:
            body = json.loads(content)

            if body.get("err_code"):
                asr_output = body.get("err_msg")
                status = TranscriptionStatus.FAILED
                _debug_log(
                    debug,
                    self.provider_name,
                    f"API returned err_code={body.get('err_code')}: "
                    f"{body.get('err_msg')}",
                )
            else:
                asr_output = DeepgramOutput.from_json(body)
                status = TranscriptionStatus.COMPLETED

        return status, asr_output

    async def result_to_dialogue(self, asr_output: DeepgramOutput) -> List[str]:
        """Convert the result to dialogue format for WER."""
        utterances: List[DeepgramUtterance] = asr_output.results.utterances

        dialogue_lines: List[str] = []
        for utterance in utterances:
            dialogue_lines.append(utterance.transcript)

        return dialogue_lines

    async def result_to_rttm(self, asr_output: DeepgramOutput) -> List[str]:
        """Convert the result to RTTM format for DER."""
        utterances: List[DeepgramUtterance] = asr_output.results.utterances

        rttm_lines: List[str] = []
        for utterance in utterances:
            start_seconds: float = utterance.start
            end_seconds: float = utterance.end
            speaker: int = utterance.speaker

            rttm_lines.append(f"{start_seconds} {end_seconds} {speaker}")

        return rttm_lines


class SubQ(Deepgram):
    """ASR provider for the subQ STT API (Deepgram-compatible).

    The SubQ sync endpoint enforces a per-request diarization budget of
    ~55s of processing time (≈ ~50 minutes of audio at typical RT factors)
    and is fronted by a proxy that returns 504 on long requests. Files
    longer than :attr:`_chunk_duration_s` are transparently split into
    WAV chunks, transcribed sequentially, and stitched back into a single
    Deepgram-shaped response with global time offsets and per-chunk
    speaker namespacing.
    """

    _chunk_duration_s: float = 1500.0
    _chunk_overlap_s: float = 0.0
    _chunk_max_retries: int = 3

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        super().__init__(api_url, api_key, options, concurrency_limit)
        self.provider_name = "subq"

    async def get_transcription(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
        debug: bool = False,
    ) -> Tuple["TranscriptionStatus", DeepgramOutput]:
        """Run the SubQ ASR provider, chunking long files when necessary."""
        duration_s = self._read_audio_duration(audio_file)

        if (
            duration_s is None
            or duration_s <= self._chunk_duration_s
            or audio_file.suffix.lower() != ".wav"
        ):
            return await super().get_transcription(audio_file, url, session, debug)

        _debug_log(
            debug,
            self.provider_name,
            f"'{audio_file.name}' duration={duration_s:.1f}s exceeds chunk "
            f"threshold ({self._chunk_duration_s}s); chunking and stitching.",
        )
        return await self._transcribe_chunked(
            audio_file, url, session, duration_s, debug
        )

    @staticmethod
    def _read_audio_duration(audio_file: Path) -> Union[float, None]:
        """Return the audio duration in seconds, or ``None`` if unavailable."""
        try:
            import soundfile as sf  # type: ignore[import-untyped]

            info = sf.info(str(audio_file))
            return float(info.frames) / float(info.samplerate)
        except Exception:
            return None

    async def _transcribe_chunked(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
        duration_s: float,
        debug: bool,
    ) -> Tuple["TranscriptionStatus", DeepgramOutput]:
        """Split ``audio_file`` into WAV chunks, transcribe each, and merge."""
        chunks = await asyncio.to_thread(
            self._chunk_wav_file, audio_file, self._chunk_duration_s
        )

        chunk_outputs: List[DeepgramOutput] = []
        chunk_starts: List[float] = []

        for chunk_idx, (start_s, chunk_bytes) in enumerate(chunks):
            _debug_log(
                debug,
                self.provider_name,
                f"'{audio_file.name}' chunk {chunk_idx + 1}/{len(chunks)} "
                f"(start={start_s:.1f}s, bytes={len(chunk_bytes)}); POSTing.",
            )
            output = await self._post_chunk_with_retry(
                chunk_bytes=chunk_bytes,
                url=url,
                session=session,
                debug=debug,
                audio_name=audio_file.name,
                chunk_idx=chunk_idx,
                total_chunks=len(chunks),
            )
            chunk_outputs.append(output)
            chunk_starts.append(start_s)

        merged = self._merge_chunk_outputs(chunk_outputs, chunk_starts, duration_s)
        return TranscriptionStatus.COMPLETED, merged

    @staticmethod
    def _chunk_wav_file(
        audio_file: Path, chunk_duration_s: float
    ) -> List[Tuple[float, bytes]]:
        """Read ``audio_file`` and return a list of ``(start_seconds, wav_bytes)``."""
        import soundfile as sf  # type: ignore[import-untyped]

        chunks: List[Tuple[float, bytes]] = []
        with sf.SoundFile(str(audio_file)) as src:
            samplerate = src.samplerate
            subtype = src.subtype
            total_frames = src.frames
            chunk_frames = max(1, int(chunk_duration_s * samplerate))

            offset = 0
            while offset < total_frames:
                src.seek(offset)
                frames = min(chunk_frames, total_frames - offset)
                data = src.read(frames=frames, dtype="int16", always_2d=True)

                buf = io.BytesIO()
                with sf.SoundFile(
                    buf,
                    mode="w",
                    samplerate=samplerate,
                    channels=data.shape[1],
                    format="WAV",
                    subtype=subtype if subtype else "PCM_16",
                ) as dst:
                    dst.write(data)

                start_s = offset / float(samplerate)
                chunks.append((start_s, buf.getvalue()))
                offset += chunk_frames

        return chunks

    async def _post_chunk_with_retry(
        self,
        chunk_bytes: bytes,
        url: HttpUrl,
        session: aiohttp.ClientSession,
        debug: bool,
        audio_name: str,
        chunk_idx: int,
        total_chunks: int,
    ) -> DeepgramOutput:
        """POST a single WAV chunk and parse the response, retrying on 504."""
        headers = {
            "Authorization": f"Token {self.config.api_key.get_secret_value()}",
            "Content-Type": "audio/wav",
        }
        _url = f"{url}{build_query_string(self.options)}"

        last_error: Union[Exception, None] = None
        for attempt in range(1, self._chunk_max_retries + 1):
            try:
                async with session.post(
                    url=_url, data=chunk_bytes, headers=headers
                ) as response:
                    if response.status == 200:
                        body = json.loads((await response.text()).strip())
                        return DeepgramOutput.from_json(body)
                    if response.status == 504:
                        raise GatewayTimeoutError(response.status)
                    raise Exception(await response.text())
            except GatewayTimeoutError as e:
                last_error = e
                _debug_log(
                    debug,
                    self.provider_name,
                    f"'{audio_name}' chunk {chunk_idx + 1}/{total_chunks} "
                    f"504 (attempt {attempt}/{self._chunk_max_retries}); retrying.",
                )
                if attempt >= self._chunk_max_retries:
                    break
                await asyncio.sleep(min(2 ** (attempt - 1), 8))

        raise last_error if last_error is not None else Exception(
            f"Failed to POST chunk {chunk_idx + 1}/{total_chunks} for {audio_name}"
        )

    @staticmethod
    def _merge_chunk_outputs(
        chunk_outputs: List[DeepgramOutput],
        chunk_starts: List[float],
        total_duration: float,
    ) -> DeepgramOutput:
        """Merge per-chunk Deepgram outputs into a single recording-level output.

        Time offsets are added to every word/utterance start/end. Speaker IDs
        are namespaced per chunk (offset by the running max + 1) because the
        sync endpoint clusters speakers per-request and there is no acoustic
        information available here to re-cluster across chunks.
        """
        merged_utterances: List[DeepgramUtterance] = []
        merged_words_by_channel: Dict[int, List[DeepgramWords]] = {}
        merged_transcript_by_channel: Dict[int, List[str]] = {}
        merged_confidence_by_channel: Dict[int, List[float]] = {}
        speaker_offset = 0

        for output, t0 in zip(chunk_outputs, chunk_starts):
            chunk_max_speaker = -1

            for utt in output.results.utterances or []:
                new_words = [
                    SubQ._shift_word(w, t0, speaker_offset) for w in utt.words
                ]
                new_utt = utt.model_copy(
                    update={
                        "start": utt.start + t0,
                        "end": utt.end + t0,
                        "speaker": utt.speaker + speaker_offset,
                        "words": new_words,
                    }
                )
                merged_utterances.append(new_utt)
                chunk_max_speaker = max(chunk_max_speaker, utt.speaker)
                for w in utt.words:
                    chunk_max_speaker = max(chunk_max_speaker, w.speaker)

            for ch_idx, channel in enumerate(output.results.channels):
                bucket_words = merged_words_by_channel.setdefault(ch_idx, [])
                bucket_text = merged_transcript_by_channel.setdefault(ch_idx, [])
                bucket_conf = merged_confidence_by_channel.setdefault(ch_idx, [])
                for alt in channel.alternatives:
                    if alt.transcript:
                        bucket_text.append(alt.transcript)
                    bucket_conf.append(alt.confidence)
                    for w in alt.words:
                        bucket_words.append(SubQ._shift_word(w, t0, speaker_offset))
                        chunk_max_speaker = max(chunk_max_speaker, w.speaker)

            speaker_offset += chunk_max_speaker + 1 if chunk_max_speaker >= 0 else 0

        channels = [
            DeepgramChannel(
                alternatives=[
                    DeepgramAlternative(
                        confidence=(
                            sum(merged_confidence_by_channel[ch_idx])
                            / len(merged_confidence_by_channel[ch_idx])
                            if merged_confidence_by_channel[ch_idx]
                            else 0.0
                        ),
                        transcript=" ".join(
                            merged_transcript_by_channel.get(ch_idx, [])
                        ).strip(),
                        words=merged_words_by_channel.get(ch_idx, []),
                    )
                ]
            )
            for ch_idx in sorted(merged_words_by_channel.keys())
        ]

        first = chunk_outputs[0]
        metadata = first.metadata.model_copy(update={"duration": total_duration})

        return DeepgramOutput(
            metadata=metadata,
            results=DeepgramResult(
                channels=channels,
                utterances=merged_utterances or None,
            ),
        )

    @staticmethod
    def _shift_word(
        word: DeepgramWords, t0: float, speaker_offset: int
    ) -> DeepgramWords:
        return word.model_copy(
            update={
                "start": word.start + t0,
                "end": word.end + t0,
                "speaker": word.speaker + speaker_offset,
            }
        )

    async def result_to_dialogue(self, asr_output: DeepgramOutput) -> List[str]:
        """Convert the result to dialogue format for WER."""
        if asr_output.results.utterances:
            return await super().result_to_dialogue(asr_output)

        dialogue_lines: List[str] = []
        for channel in asr_output.results.channels:
            for alt in channel.alternatives:
                if alt.transcript:
                    dialogue_lines.append(alt.transcript)
        return dialogue_lines

    async def result_to_rttm(self, asr_output: DeepgramOutput) -> List[str]:
        """Convert the result to RTTM format for DER."""
        if asr_output.results.utterances:
            return await super().result_to_rttm(asr_output)

        rttm_lines: List[str] = []
        for channel in asr_output.results.channels:
            for alt in channel.alternatives:
                for word in alt.words:
                    rttm_lines.append(f"{word.start} {word.end} {word.speaker}")
        return rttm_lines


class Google(ASRProvider):
    """The ASR provider class for Google."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = GoogleOptions(**options)

    @property
    def output_schema(self) -> GoogleOutput:
        """The output format of the Google ASR provider."""
        return GoogleOutput

    async def get_transcription(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
        debug: bool = False,
    ) -> Tuple[TranscriptionStatus, GoogleOutput]:
        """Run the ASR provider."""
        raise NotImplementedError("Google not implemented.")

    async def result_to_dialogue(self, asr_output: GoogleOutput) -> List[str]:
        """Convert the result to dialogue format for WER."""
        pass

    async def result_to_rttm(self, asr_output: GoogleOutput) -> List[str]:
        """Convert the result to RTTM format for DER."""
        pass


class RevAI(ASRProvider):
    """The ASR provider class for RevAI."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = RevAIOptions(**options)

    @property
    def output_schema(self) -> RevAIOutput:
        """The output format of the RevAI ASR provider."""
        return RevAIOutput

    async def get_transcription(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
        debug: bool = False,
    ) -> Tuple[TranscriptionStatus, RevAIOutput]:
        """Call the API of the RevAI ASR provider."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
        }

        async with aiofiles.open(audio_file, mode="rb") as f:
            form = aiohttp.FormData()
            form.add_field("media", f, filename=audio_file.name)
            form.add_field("options", json.dumps(self.options, sort_keys=True))

            async with session.post(
                url=f"{url}/jobs/", data=form, headers=headers
            ) as response:
                if response.status == 200:
                    content = (await response.text()).strip()
                elif response.status == 504:
                    raise GatewayTimeoutError(response.status)
                else:
                    raise Exception(await response.text())

        body = json.loads(content)
        job_id = body.get("id")

        while True:
            async with session.get(
                url=f"{url}/jobs/{job_id}", headers=headers
            ) as response:
                if response.status == 200:
                    content = (await response.text()).strip()

                    body = json.loads(content)
                    if body.get("status") == "transcribed":
                        status = TranscriptionStatus.COMPLETED
                        break
                    elif body.get("status") == "failed":
                        status = TranscriptionStatus.FAILED
                        break
                    else:
                        await asyncio.sleep(3)
                elif response.status == 504:
                    await asyncio.sleep(3)
                else:
                    raise Exception(await response.text())

        if status == TranscriptionStatus.COMPLETED:
            async with session.get(
                url=f"{url}/jobs/{job_id}/transcript", headers=headers
            ) as response:
                if response.status == 200:
                    content = (await response.text()).strip()
                else:
                    raise Exception(await response.text())

            body = json.loads(content)
            asr_output = RevAIOutput.from_json(body)

        else:
            asr_output = body.get("failure_detail")

        return status, asr_output

    async def result_to_dialogue(self, asr_output: RevAIOutput) -> List[str]:
        """Convert the result to dialogue format for WER."""
        monologues: List[RevAIMonologue] = asr_output.monologues

        dialogue_lines: List[str] = []
        for monologue in monologues:
            elements: List[RevAIElement] = monologue.elements
            text = "".join([element.value for element in elements])

            dialogue_lines.append(text.strip())

        return dialogue_lines

    async def result_to_rttm(self, asr_output: RevAIOutput) -> List[str]:
        """Convert the result to RTTM format."""
        monologues: List[RevAIMonologue] = asr_output.monologues

        rttm_lines: List[str] = []
        for monologue in monologues:
            start_seconds, end_seconds = self._get_timestamps(
                elements=monologue.elements
            )
            speaker: int = monologue.speaker

            rttm_lines.append(f"{start_seconds} {end_seconds} {speaker}")

        return rttm_lines

    @staticmethod
    def _get_timestamps(elements: List[RevAIElement]) -> Tuple[float, float]:
        """Retrieve the start and end timestamps of a monologue."""
        start_ts = 0
        end_ts = 0

        for i in range(len(elements)):
            if elements[i].type == "text":
                start_ts = elements[i].ts
                break

        for i in range(len(elements) - 1, -1, -1):
            if elements[i].type == "text":
                end_ts = elements[i].end_ts
                break

        return start_ts, end_ts


class Speechmatics(ASRProvider):
    """The ASR provider class for Speechmatics."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = SpeechmaticsOptions(**options)

    @property
    def output_schema(self) -> SpeechmaticsOutput:
        """The output format of the Speechmatics ASR provider."""
        return SpeechmaticsOutput

    async def get_transcription(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
        debug: bool = False,
    ) -> Tuple[TranscriptionStatus, SpeechmaticsOutput]:
        """Call the API of the Speechmatics ASR provider."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
        }

        _url = f"{url}/jobs"
        async with aiofiles.open(audio_file, mode="rb") as f:
            form = aiohttp.FormData()
            form.add_field("data_file", f, filename=audio_file.name)
            form.add_field("config", json.dumps(self.options, ensure_ascii=False))

            async with session.post(url=_url, data=form, headers=headers) as response:
                if response.status == 201:
                    content = (await response.text()).strip()
                elif response.status == 503:
                    raise GatewayTimeoutError(response.status)
                else:
                    raise Exception(await response.text())

        body = json.loads(content)
        if body.get("error") == "Forbidden":
            status = TranscriptionStatus.FAILED
            asr_output = Exception(body.get("detail"))
        else:
            job_id = body.get("id")

        await asyncio.sleep(1)  # Extra pause for Speechmatics

        while True:
            async with session.get(url=f"{_url}/{job_id}", headers=headers) as response:
                if response.status == 200:
                    content = (await response.text()).strip()
                elif response.status == 503:
                    asyncio.sleep(3)
                else:
                    raise Exception(await response.text())

            body = json.loads(content)
            _job = body.get("job")

            if _job.get("status") == "done":
                status = TranscriptionStatus.COMPLETED
                break
            elif _job.get("status") == "rejected":
                status = TranscriptionStatus.FAILED
                break
            else:
                await asyncio.sleep(3)

        if status == TranscriptionStatus.COMPLETED:
            async with session.get(
                url=f"{_url}/{job_id}/transcript?format=json-v2",
                headers=headers,
            ) as response:
                if response.status == 200:
                    content = (await response.text()).strip()
                elif response.status == 503:
                    raise GatewayTimeoutError(response.status)
                else:
                    raise Exception(await response.text())

            body = json.loads(content)
            asr_output = SpeechmaticsOutput.from_json(body)

        else:
            _errors = body.get("errors")
            asr_output = "\n".join([error.get("message") for error in _errors])

        return status, asr_output

    async def result_to_dialogue(self, asr_output: SpeechmaticsOutput) -> List[str]:
        """Convert the result to dialogue format for WER."""
        results: List[SpeechmaticsResult] = asr_output.results

        dialogue_lines: List[str] = []
        text = ""
        for result in results:
            _content = result.alternatives[0].content

            if result.type == "word":
                text += f" {_content}"

            elif result.type == "punctuation":
                if result.attaches_to == "previous":
                    text += f"{_content}"
                else:
                    text += f" {_content}"

        dialogue_lines.append(text.strip())

        return dialogue_lines

    async def result_to_rttm(self, asr_output: SpeechmaticsOutput) -> List[str]:
        """Convert the result to RTTM format for DER."""
        results: List[SpeechmaticsResult] = asr_output.results

        rttm_lines: List[str] = []
        for result in results:
            if result.type == "word":
                start_seconds: float = result.start_time
                end_seconds: float = result.end_time
                speaker: str = result.alternatives[0].speaker

                if speaker != "UU":
                    rttm_lines.append(f"{start_seconds} {end_seconds} {speaker}")

        return rttm_lines


class Wordcab(ASRProvider):
    """The ASR provider class for Wordcab."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        """Initialize the Wordcab ASR provider."""
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = WordcabOptions(**options)

    @property
    def output_schema(self) -> WordcabOutput:
        """The output format of the Wordcab ASR provider."""
        return WordcabOutput

    async def get_transcription(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
        debug: bool = False,
    ) -> Tuple[str, TranscriptionStatus, WordcabOutput]:
        """Run the Wordcab ASR provider."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
            "Content-Disposition": f'attachment; filename="{audio_file.name}"',
        }

        _url = f"{url}/transcribe{build_query_string(self.options)}"
        async with aiofiles.open(audio_file, mode="rb") as f:
            form = aiohttp.FormData()
            form.add_field("file", f, filename=audio_file.name)

            async with session.post(url=_url, data=form, headers=headers) as response:
                if response.status == 201 or response.status == 200:
                    content = (await response.text()).strip()
                elif response.status == 504:
                    raise GatewayTimeoutError(response.status)
                else:
                    raise Exception(f"Wordcab API unavailable {response.status}.")

        body = json.loads(content)
        job_name = body.get("job_name")
        transcript_id = body.get("transcript_id")

        while True:
            async with session.get(
                url=f"{url}/jobs/{job_name}", headers=headers
            ) as response:
                if response.status == 200:
                    content = (await response.text()).strip()
                elif response.status == 504:
                    asyncio.sleep(3)
                else:
                    raise Exception(await response.text())

            body = json.loads(content)
            if body.get("job_status") == "TranscriptComplete":
                status = TranscriptionStatus.COMPLETED
                break
            elif body.get("job_status") == "Error":
                status = TranscriptionStatus.FAILED
                break
            else:
                await asyncio.sleep(3)

        if status == TranscriptionStatus.COMPLETED:
            async with session.get(
                url=f"{url}/transcripts/{transcript_id}", headers=headers
            ) as response:
                if response.status == 200:
                    content = (await response.text()).strip()
                elif response.status == 504:
                    raise GatewayTimeoutError(response.status)
                else:
                    raise Exception(await response.text())

            body = json.loads(content)
            asr_output = WordcabOutput.from_json(body)

        else:
            asr_output = body.get("error_message")

        return status, asr_output

    async def result_to_dialogue(self, asr_output: WordcabOutput) -> List[str]:
        """Convert the result to dialogue format for WER."""
        all_transcripts: List[WordcabTranscript] = asr_output.transcript

        dialogue_lines: List[str] = []
        for transcript in all_transcripts:
            dialogue_lines.append(transcript.text)

        return dialogue_lines

    async def result_to_rttm(self, asr_output: WordcabOutput) -> List[str]:
        """Convert the result to RTTM format for DER."""
        all_transcripts: List[WordcabTranscript] = asr_output.transcript

        rttm_lines: List[str] = []
        for transcript in all_transcripts:
            start_seconds: float = transcript.timestamp_start / 1000
            end_seconds: float = transcript.timestamp_end / 1000
            speaker: str = transcript.speaker

            rttm_lines.append(f"{start_seconds} {end_seconds} {speaker}")

        return rttm_lines


class WordcabHosted(ASRProvider):
    """The ASR provider class for Wordcab hosted."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
        host: str,
        port: int,
    ) -> None:
        _api_url = api_url.format(host=host, port=port)
        super().__init__(_api_url, api_key, concurrency_limit)
        self.options = WordcabHostedOptions(**options)
        self.provider_name = "wordcab-hosted"

    @property
    def output_schema(self) -> WordcabHostedOutput:
        """ "The output format of the Wordcab hosted ASR provider."""
        return WordcabHostedOutput

    async def get_transcription(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
        debug: bool = False,
    ) -> Tuple[str, TranscriptionStatus, WordcabHostedOutput]:
        """Run the Wordcab hosted ASR provider."""
        async with aiofiles.open(audio_file, mode="rb") as f:
            form = aiohttp.FormData()
            form.add_field("file", f, filename=audio_file.name)

            for k, v in self.options.items():
                if isinstance(v, (dict, list, tuple)):
                    serialized_value = json.dumps(v)
                else:
                    serialized_value = str(v)

                form.add_field(k, serialized_value)

            async with session.post(url=str(url), data=form) as response:
                if response.status == 201 or response.status == 200:
                    content = (await response.text()).strip()
                elif response.status == 504:
                    raise GatewayTimeoutError(response.status)
                else:
                    raise Exception(
                        f"Wordcab Hosted API unavailable {response.status}."
                    )

        body = json.loads(content)
        asr_output = WordcabHostedOutput.from_json(body)

        return TranscriptionStatus.COMPLETED, asr_output

    async def result_to_dialogue(self, asr_output: WordcabHostedOutput) -> List[str]:
        """Convert the result to dialogue format for WER."""
        all_transcripts: List[WordcabHostedTranscript] = asr_output.utterances

        dialogue_lines: List[str] = []
        for transcript in all_transcripts:
            dialogue_lines.append(transcript.text)

        return dialogue_lines

    async def result_to_rttm(self, asr_output: WordcabHostedOutput) -> List[str]:
        """Convert the result to RTTM format for DER."""
        all_transcripts: List[WordcabHostedTranscript] = asr_output.utterances

        rttm_lines: List[str] = []
        for transcript in all_transcripts:
            start_seconds: float = transcript.start
            end_seconds: float = transcript.end
            speaker: str = transcript.speaker

            rttm_lines.append(f"{start_seconds} {end_seconds} {speaker}")

        return rttm_lines
