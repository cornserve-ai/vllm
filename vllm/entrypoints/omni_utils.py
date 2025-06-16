import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any, Union

from transformers import AutoProcessor

from cornserve.task_executors.eric.api import Modality
from cornserve.task_executors.eric.config import ModalityConfig
from cornserve.task_executors.eric.router.processor import ModalityDataLoader
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

logger = init_logger(__name__)
thread_local = threading.local()


class OmniProcessor:
    """Async thread pool processor for Omni model."""
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Omni-7B", max_workers: int = 4) -> None:

        def init_thread() -> None:
            """Initialize thread-local processor and loader."""
            thread_local.processor = AutoProcessor.from_pretrained(model_name)
            thread_local.loader = ModalityDataLoader(ModalityConfig())

        self.loop = asyncio.get_event_loop()
        self.pool = ThreadPoolExecutor(max_workers=max_workers, initializer=init_thread)

    async def process(self, messages: list[ChatCompletionMessageParam]) -> TextPrompt:
        return await self.loop.run_in_executor(self.pool, self._do_process, messages)

    def _do_process(self, messages: list[ChatCompletionMessageParam]) -> TextPrompt:
        loader = thread_local.loader
        processor = thread_local.processor

        messages, image_urls, video_urls, audio_urls = convert_messages(messages)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        video_data = [loader.load_from_url(Modality.VIDEO, u) for u in video_urls]
        image_data = [loader.load_from_url(Modality.IMAGE, u) for u in image_urls]
        audio_data = [loader.load_from_url(Modality.AUDIO, u) for u in audio_urls]
        mm_data: dict[str, Any] = {"use_audio_in_video": False}
        if len(video_data) > 0:
            mm_data["video"] = video_data
        if len(image_data) > 0:
            mm_data["image"] = image_data
        if len(audio_data) > 0:
            mm_data["audio"] = audio_data

        return TextPrompt(prompt=prompt[0], multi_modal_data=mm_data)

def convert_messages(chat_messages: Union[list, dict]) -> tuple[list, list, list, list]:
    """
    Convert chat_messages format to messages format.
    
    Args:
        chat_messages: List of ChatCompletionMessageParam
        
    Returns:
        Tuple containing:
        - List of message dictionaries in messages format
        - List of image URLs
        - List of video URLs  
        - List of audio URLs
    """
    messages = []
    image_urls = []
    video_urls = []
    audio_urls = []
    
    for message in chat_messages:
        # Copy the message structure
        converted_message = {
            "role": message["role"],
            "content": []
        }
        
        # Process each content item
        for content_item in message["content"]:
            if content_item["type"] == "image_url":
                # Convert image_url to image format
                url = content_item["image_url"]
                converted_message["content"].append({
                    "type": "image",
                    "url": url
                })
                image_urls.append(url)
            elif content_item["type"] == "video_url":
                # Convert video_url to video format
                url = content_item["video_url"]
                converted_message["content"].append({
                    "type": "video",
                    "url": url
                })
                video_urls.append(url)
            elif content_item["type"] == "audio_url":
                # Convert audio_url to audio format
                url = content_item["audio_url"]
                converted_message["content"].append({
                    "type": "audio",
                    "url": url
                })
                audio_urls.append(url)
            else:
                # Keep other types as-is (like text)
                converted_message["content"].append(content_item)
        
        messages.append(converted_message)
    
    return messages, image_urls, video_urls, audio_urls


def make_inputs_qwen2_omni(
    chat_messages: list[ChatCompletionMessageParam],
    model_name: str = "Qwen/Qwen2.5-Omni-7B"
) -> TextPrompt:
    processor = AutoProcessor.from_pretrained(model_name)

    messages, image_urls, video_urls, audio_urls = convert_messages(chat_messages)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


    loader = ModalityDataLoader(ModalityConfig())
    video_data = [loader.load_from_url(Modality.VIDEO, u) for u in video_urls]
    image_data = [loader.load_from_url(Modality.IMAGE, u) for u in image_urls]
    audio_data = [loader.load_from_url(Modality.AUDIO, u) for u in audio_urls]

    mm_data : dict[str, Any] = {"use_audio_in_video": False}
    if len(video_data) > 0:
        mm_data["video"] = video_data
    if len(image_data) > 0:
        mm_data["image"] = image_data
    if len(audio_data) > 0:
        mm_data["audio"] = audio_data

    return TextPrompt(
        prompt=prompt[0],
        multi_modal_data=mm_data,
    )
