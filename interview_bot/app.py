import asyncio
import pathlib
from typing import AsyncGenerator, Literal
import json
from datetime import datetime
from google import genai
from google.genai import types
from google.genai.types import (
    Content,
    LiveConnectConfig,
    Part,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
)
import boto3
import os
from dotenv import load_dotenv
import gradio as gr
from fastrtc import AsyncStreamHandler, WebRTC, async_aggregate_bytes_to_16bit
import numpy as np
import validate_code
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TRANSCRIPT_BUCKET = os.getenv("S3_BUCKET", "interview-transcripts-bucket")
session = boto3.Session(profile_name=os.environ["AWS_PROFILE"])
s3 = session.client("s3", region_name="us-east-1")
load_dotenv()
current_dir = pathlib.Path(__file__).parent

class GeminiHandler(AsyncStreamHandler):
    """Handler for the Gemini API"""

    def __init__(
        self,
        expected_layout: Literal["mono"] = "mono",
        output_sample_rate: int = 24000,
        output_frame_size: int = 480,
        input_sample_rate: int = 16000,
    ) -> None:
        super().__init__(
            expected_layout,
            output_sample_rate,
            output_frame_size,
            input_sample_rate=input_sample_rate,
        )
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.quit: asyncio.Event = asyncio.Event()
        self.transcripts = []
        self.current_gemini_utterance = ""
        self.current_user_utterance = ""
        self.session_resume_handle: str | None = None 


    def copy(self) -> "GeminiHandler":
        """Required implementation of the copy method for AsyncStreamHandler"""
        return GeminiHandler(
            expected_layout=self.expected_layout,
            output_sample_rate=self.output_sample_rate,
            output_frame_size=self.output_frame_size,
        )

    async def stream(self) -> AsyncGenerator[bytes, None]:
        """Helper method to stream input audio to the server. Used in start_stream."""
        while not self.quit.is_set():
                audio = await asyncio.wait_for(self.input_queue.get(), timeout=30)
                yield audio
        return

    async def connect(
        self,
        project_id: str,
        location: str,
        voice_name: str | None = None,
        system_instruction: str | None = None,
        resume_handle: str | None = None
    ) -> AsyncGenerator[bytes, None]:
        """Connect to the Gemini server and start the stream."""
        
        client = genai.Client(vertexai=True, project=project_id, location=location)
        config = LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription={},  
            input_audio_transcription={},
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
            ),
            system_instruction=Content(parts=[Part.from_text(text=system_instruction)]),
            session_resumption=types.SessionResumptionConfig(
                handle=resume_handle
            )
        )
        async with client.aio.live.connect(
            model="gemini-2.0-flash-live-preview-04-09", config=config
        ) as session:
            async for audio in session.start_stream(
                stream=self.stream(), mime_type="audio/pcm"
            ):
                if audio.data:
                    yield audio.data

                # if audio.usage_metadata:
                #     usage = audio.usage_metadata
                #     logger.info(f" Used {usage.total_token_count} tokens in total.")
                #     for detail in usage.response_tokens_details:
                #         match detail:
                #             case types.ModalityTokenCount(modality=modality, token_count=count):
                #                 logger.info(f"    {modality}: {count} tokens")

                if audio.server_content:
                    if audio.server_content.output_transcription:
                        text = audio.server_content.output_transcription.text

                        if self.current_user_utterance:
                            self.transcripts.append({
                                "speaker": "user",
                                "text": self.current_user_utterance.strip()
                            })
                            self.current_user_utterance = ""

                        if text:
                            self.current_gemini_utterance += text
                        elif self.current_gemini_utterance:
                            self.transcripts.append({
                                "speaker": "gemini",
                                "text": self.current_gemini_utterance.strip()
                            })
                            self.current_gemini_utterance = ""

                    if audio.server_content.input_transcription:
                        text = audio.server_content.input_transcription.text
                        if text:
                            self.current_user_utterance += text
                if audio.session_resumption_update:  
                    update = audio.session_resumption_update
                    if update.resumable and update.new_handle:
                        self.session_resume_handle = update.new_handle
                        logger.info(f"Received resumable session handle: {update.new_handle}")
                        (current_dir / "resume_handle.txt").write_text(update.new_handle)

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """Receive audio from the user and put it in the input stream."""
        _, array = frame
        array = array.squeeze()
        audio_message = array.tobytes()
        self.input_queue.put_nowait(audio_message)

    async def generator(self) -> None:
        """Helper method for placing audio from the server into the output queue."""
        project_id = os.environ['PROJECT_ID']
        location = os.environ['LOCATION']
        voice_name = os.environ['VOICE_NAME']
        system_instruction = """
        ## ROLE & OBJECTIVE
        You are an Expert AI Interviewer. Your objective is to conduct a focused  conceptual interview with a candidate. The interview will cover:
        1. Classical Machine Learning
        2. Neural Networks and Transformers 
        3. Retrieval-Augmented Generation (RAG)
        4. A recent project
        5. Past organizational experience

        ## INTERVIEW STRUCTURE
        ### 1. Introduction
        - Briefly introduce yourself.
        - Inform the candidate that the interview will cover the technical questions.
        - Let the candidate know they are welcome to ask for the question to be repeated or to say that they didn't understand or don't know the answer.
        - Ask the candidate if they are ready to begin and transition into the first technical question (start with Classical Machine Learning).
        - Tell the candidate that this interview is for the role of an AI engineer intern at cognito labs if asked.

        ### 2. Technical Topics
        - Ask **one conceptual question at a time** for each of the following:
        - **Classical Machine Learning:** Ask multiple in-depth questions (approximately 7–8). Probe the candidate’s understanding of core ML concepts.
        - **Transformers:** Ask beginner-friendly questions only. Avoid in-depth or highly technical questions in this section.
        - **Retrieval-Augmented Generation (RAG):** Ask only beginner-level questions. Do not ask deep technical or implementation-level questions for this topic.
        - Ask thoughtful follow-up questions based on the candidate's responses to probe depth of understanding.
        - Ask multiple questions for each topic.

        ### 3. Recent Project 
        - Ask the candidate to describe a recent project they worked on.
        - Ask follow-up questions focusing on:
        - Techniques and tools used, Metrics, evaluation
        - Challenges and how they were addressed

        ### 4. Past Internship/Organizational Experience
        - Ask about a prior role or organization.
        - Probe into:
        - Responsibilities
        - Major technical challenges
        - Key learnings or takeaways

        ### 5. Conclusion (Brief)
        - End politely and clearly once all topics have been covered.
        - If the candidate talks in any other language or asks about things irrelevant even after the interview, refrain from discussing.

        ## CONVERSATION RULES
        - Ask One Question at a time. Always wait for the candidate’s full response before continuing.
        - If the candidate hasn’t answered the current question, do not proceed. give enough time to answer before asking the next question.
        - Do not explain answers, if the answer is incomplete or incorrect.
        - Listen carefully to the candidate’s response before proceeding to the next question. Avoid interrupting or switching questions while they are speaking. Allow sufficient time for them to answer fully
        - After each answer, briefly acknowledge the response and proceed to the next question. Do not answer the questions you have asked.
        - If the candidate veers off-topic, redirect politely: “Let’s focus on the interview topics for now.”
        - Sound conversational—not scripted. Build follow-up questions naturally based on what the candidate just said.
        - Ask multiple questions for each topic (e.g., 7–8 questions on machine learning, 7–8 on transformers, and so on).
        - Ensure all five topics are covered.
    """
        resume_handle_file = current_dir / "resume_handle.txt"
        resume_handle = None
        if resume_handle_file.exists():
            resume_handle = resume_handle_file.read_text().strip()
            logger.info(f"Resuming session from handle: {resume_handle}")
        try:
            async for audio_response in async_aggregate_bytes_to_16bit(
                self.connect(project_id, location, voice_name, system_instruction, resume_handle)
            ):
                self.output_queue.put_nowait(audio_response)
        except asyncio.TimeoutError:
            logger.warning("Frame aggregation timed out — stream likely stalled.")
        except Exception as e:
            logger.error(f"Unhandled error in audio generator: {e}")

    async def emit(self) -> tuple[int, np.ndarray]:
        """Required implementation of the emit method for AsyncStreamHandler"""
        if not self.args_set.is_set():
            await self.wait_for_args()
            asyncio.create_task(self.generator())

        array = await self.output_queue.get()
        return (self.output_sample_rate, array)

    def shutdown(self) -> None:
        """Stop the stream method on shutdown"""
        self.quit.set()
        if self.current_user_utterance:
            self.transcripts.append({
                "speaker": "user",
                "text": self.current_user_utterance.strip()
            })

        if self.current_gemini_utterance:
            self.transcripts.append({
                "speaker": "gemini",
                "text": self.current_gemini_utterance.strip()
            })

        if self.transcripts:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{validate_code.CURRENT_CODE}_{timestamp}.json"
            s3_key = f"interviewtranscripts/{filename}"
            transcript_json = json.dumps(self.transcripts, indent=2, ensure_ascii=False)
            s3.put_object(
                Bucket=TRANSCRIPT_BUCKET,
                Key=s3_key,
                Body=transcript_json.encode("utf-8"),
                ContentType="application/json"
            )
            print(f"Transcripts uploaded to s3://{TRANSCRIPT_BUCKET}/{s3_key}")

css = (current_dir / "style.css").read_text()
header = (current_dir / "header.html").read_text()
with gr.Blocks(css=css) as interview:
    gr.HTML(header)
    with gr.Column():
        one_time_id = gr.Textbox(
            label="Enter One-Time Interview Code",
            placeholder="12 digit code",
            type="text"
        )
        email_input = gr.Textbox(
            label="Enter Your Email",
            placeholder="you@example.com",
            type="text"
        )
        submit_btn = gr.Button("Start Interview")

    with gr.Row(visible=False) as interview_ui:
        with gr.Row():  
            webrtc = WebRTC(
                label="🎙️ Interview Audio Stream",
                modality="audio",
                mode="send-receive",  
                rtc_configuration=None,
            )

            webrtc.stream(
                GeminiHandler(),  
                inputs=[webrtc],
                outputs=[webrtc],
                time_limit=1000,
                concurrency_limit=2,
            )
    submit_btn.click(
    validate_code.validate_code,
    inputs=[one_time_id, email_input], 
    outputs=[one_time_id, email_input, submit_btn, interview_ui],
)

if __name__ == "__main__":
    interview.launch()