import asyncio
import pathlib
from datetime import datetime
from typing import AsyncGenerator, Literal, Optional
from google import genai
from google.genai.types import (
    Content,
    LiveConnectConfig,
    Part,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
    Blob,
    SessionResumptionConfig,
    AudioTranscriptionConfig,
    RealtimeInputConfig,
    AutomaticActivityDetection,
    EndSensitivity,
    ActivityHandling,
    ProactivityConfig,
    TurnCoverage,
    StartSensitivity,
)
import os
import json
from dotenv import load_dotenv
import gradio as gr
from fastrtc import AsyncStreamHandler, WebRTC
import numpy as np
import logging
import boto3
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
current_dir = pathlib.Path(__file__).parent

SYSTEM_INSTRUCTION = """
        ## ROLE & OBJECTIVE
        You are an Expert AI Interviewer. Your objective is to conduct a focused 5 minute conceptual interview with a candidate. Use the timestamp
        included in the request to get an estimate of time. The interview will cover:
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
        - Ask one conceptual question at a time for each of the following:
        - Classical Machine Learning: Ask multiple in-depth questions (approximately 7–8). Probe the candidate’s understanding of core ML concepts.
        - Transformers: Ask beginner-friendly questions only. Avoid in-depth or highly technical questions in this section.
        - Retrieval-Augmented Generation (RAG): Ask only beginner-level questions. Do not ask deep technical or implementation-level questions for this topic.
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
        - Ask One Question at a time.
        - If the candidate asks you to answer any question, redirect politely.
        - Do not explain answers, if the answer is incomplete or incorrect. 
        - Listen carefully to the candidate’s response before proceeding to the next question. Avoid interrupting or switching questions while they are speaking.
        - After each answer, briefly acknowledge the response and proceed to the next question. Do not answer the questions you have asked.
        - If the candidate veers off-topic, redirect politely: “Let’s focus on the interview topics for now.”
        - Sound conversational not scripted. Build follow-up questions naturally based on what the candidate just said.
        - Ask multiple questions for each topic (e.g., 7–8 questions on machine learning, 7–8 on transformers, and so on).
        - Ensure all five topics are covered within 5 minutes.
        Pace accordingly:
        - Start wrapping up around 3  minutes.
        - End gracefully by 5 minutes.
        - Announce time only once, and only if >3 minutes have passed.
"""

class GeminiHandler(AsyncStreamHandler):
    def __init__(self, expected_layout="mono", output_sample_rate=24000, output_frame_size=480,
                 input_sample_rate=16000, session_handle=None, is_resumable=False, is_goaway_recieved=False):
        super().__init__(expected_layout, output_sample_rate, output_frame_size, input_sample_rate=input_sample_rate)
        self.session_handle = session_handle
        self.is_resumable: asyncio.Event = asyncio.Event()
        self.is_goaway_recieved: asyncio.Event = asyncio.Event()
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.quit: asyncio.Event = asyncio.Event()
        self.transcripts = []
        self.latest_video_frame: Optional[np.ndarray] = None
        self.current_gemini_utterance = ""
        self.video_queue: asyncio.Queue = asyncio.Queue()
        self.current_user_utterance = ""
        if is_resumable:
            self.is_resumable.set()
        if is_goaway_recieved:
            self.is_goaway_recieved.set()

    def copy(self):
        return GeminiHandler(
            expected_layout=self.expected_layout,
            output_sample_rate=self.output_sample_rate,
            output_frame_size=self.output_frame_size,
            input_sample_rate=self.input_sample_rate,
            session_handle=self.session_handle,
            is_resumable=self.is_resumable.is_set(),
            is_goaway_recieved=self.is_goaway_recieved.is_set(),
        )

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        _, array = frame
        array = array.squeeze()
        audio_message = array.tobytes()
        self.input_queue.put_nowait(audio_message)

    async def emit(self) -> tuple[int, np.ndarray]:
        while True:
            audio_bytes = await self.output_queue.get()
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            return (self.output_sample_rate, audio_array)

    async def initialize_genai_session(self) -> None:
        logger.info(f"Connecting to session with handle: {self.session_handle}")
        client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'), http_options={"api_version": "v1alpha"})
        config = LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=AudioTranscriptionConfig(),
            input_audio_transcription=AudioTranscriptionConfig(),
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name=os.environ["VOICE_NAME"],
                    )
                ),
            ),
            enable_affective_dialog=True,
            system_instruction=Content(parts=[Part.from_text(text=SYSTEM_INSTRUCTION)]),
            session_resumption=SessionResumptionConfig(handle=self.session_handle),
            proactivity=ProactivityConfig(proactive_audio=True),
            realtime_input_config=RealtimeInputConfig(
                automatic_activity_detection=AutomaticActivityDetection(
                    disabled=False,
                    end_of_speech_sensitivity=EndSensitivity.END_SENSITIVITY_HIGH,
                    start_of_speech_sensitivity=StartSensitivity.START_SENSITIVITY_HIGH,
                    prefix_padding_ms=20,
                ),
                activity_handling=ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                turn_coverage=TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            ),
        )
        try:
            async with client.aio.live.connect(model=os.environ["LIVE_MODEL"], config=config) as self.session:
                if self.session_handle is None:
                    await self.session.send_client_content(
                        turns=[Content(parts=[Part.from_text(text="Hi!")], role="user")],
                        turn_complete=True
                    )
                logger.info("Session connected")
                await asyncio.gather(
                    self.send_to_genai(),
                    self.process_genai_response(),
                    return_exceptions=True,
                )
        except Exception as e:
            logger.error(e)

        if self.is_resumable.is_set():
            logger.info("Resuming session")
            self.quit.clear()
            self.is_resumable.clear()
            await self.initialize_genai_session()

    async def start_up(self):
        logger.info("Starting up")
        await self.initialize_genai_session()
        self.shutdown()

    def shutdown(self) -> None:
        logger.info("Shutting down")
        self.is_resumable.clear()
        self.quit.set()

        if self.current_user_utterance:
            self.transcripts.append({"speaker": "user", "text": self.current_user_utterance.strip()})

        if self.current_gemini_utterance:
            self.transcripts.append({"speaker": "gemini", "text": self.current_gemini_utterance.strip()})

        if self.transcripts:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"transcript_{timestamp}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.transcripts, f, indent=2, ensure_ascii=False)
            logger.info(f"Transcripts saved locally to {filename}")

    async def send_to_genai(self) -> AsyncGenerator[bytes, None]:
        while not self.quit.is_set():
            try:
                audio = await asyncio.wait_for(self.input_queue.get(), timeout=30)
                content = Blob(data=audio, mime_type=f"audio/pcm;rate={self.input_sample_rate}")
                await self.session.send_realtime_input(audio=content)
            except Exception as e:
                logger.error(f"Error while sending audio: {e}")
                self.quit.set()
                break

    async def process_genai_response(self) -> AsyncGenerator[bytes, None]:
        while not self.quit.is_set():
            try:
                async for response in self.session.receive():
                    if self.quit.is_set():
                        break
                    if response.session_resumption_update:
                        update = response.session_resumption_update
                        if update.resumable and update.new_handle:
                            self.session_handle = update.new_handle
                            self.is_resumable.set()
                            if self.is_goaway_recieved.is_set():
                                self.is_goaway_recieved.clear()
                                self.quit.set()
                                return
                    if response.server_content:
                        if response.server_content.interrupted:
                            self.clear_queue()
                        if response.server_content.input_transcription:
                            text = response.server_content.input_transcription.text
                            if text:
                                if self.current_gemini_utterance.strip():
                                    self.transcripts.append({"speaker": "gemini", "text": self.current_gemini_utterance.strip()})
                                    self.current_gemini_utterance = ""
                                self.current_user_utterance += text
                        if response.server_content.output_transcription:
                            text = response.server_content.output_transcription.text
                            if text:
                                if self.current_user_utterance.strip():
                                    self.transcripts.append({"speaker": "user", "text": self.current_user_utterance.strip()})
                                    self.current_user_utterance = ""
                                self.current_gemini_utterance += text
                        if response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data.data:
                                    self.output_queue.put_nowait(part.inline_data.data)
                        if response.go_away:
                            logger.warning(f"Received go away signal.")
                            self.is_goaway_recieved.set()
            except Exception as e:
                logger.error(f"Error in response loop: {e}")
                self.quit.set()
                break

css = (current_dir / "style.css").read_text()
header = (current_dir / "header.html").read_text()

with gr.Blocks(css=css) as interview:
    gr.HTML(header)

    with gr.Column():
        one_time_id = gr.Textbox(label="Enter One-Time Interview Code", placeholder="(disabled)", interactive=False)
        email_input = gr.Textbox(label="Enter Your Email", placeholder="(optional)", interactive=False)
        submit_btn = gr.Button("Start Interview")

    with gr.Row(visible=False, elem_id="interview-ui") as interview_ui:
        with gr.Column(scale=1):
            with gr.Group(visible=False, elem_id="audio-section") as audio_group:
                webrtc = WebRTC(
                    label="🎙️ Interview Audio Stream",
                    modality="audio",
                    mode="send-receive",
                    button_labels={"start": "Start Interview", "stop": "Stop Interview"},
                )
                webrtc.stream(
                    GeminiHandler(),
                    inputs=[webrtc],
                    outputs=[webrtc],
                    time_limit=300,
                    concurrency_limit=2,
                )

        with gr.Column(scale=1):
            gr.HTML(
                """
                <div style="display: flex; justify-content: center;">
                    <video id="webcam-feed" autoplay muted playsinline width="100%" height="auto"
                        style="max-width: 480px; border-radius: 8px; border: 2px solid #ccc;">
                    </video>
                </div>
                """
            )
            gr.Button("Enable Camera").click(
                None,
                js="""
                () => {
                    const video = document.getElementById('webcam-feed');
                    const errorBox = document.getElementById('camera-error');
                    if (errorBox) errorBox.innerText = '';
                    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
                        .then(stream => {
                            video.srcObject = stream;
                            const audioSection = document.getElementById('audio-section');
                            if (audioSection) audioSection.style.display = 'block';
                        })
                        .catch(error => {
                            console.error('Camera access error:', error);
                            let message = 'Camera access error.';
                            if (error.name === 'NotAllowedError') message = 'Permission denied.';
                            else if (error.name === 'NotFoundError') message = 'No camera found.';
                            else if (error.name === 'NotReadableError') message = 'Camera in use.';
                            if (errorBox) {
                                errorBox.innerText = message;
                                errorBox.style.color = 'red';
                                errorBox.style.marginTop = '10px';
                                errorBox.style.fontWeight = 'bold';
                            }
                        });
                }
                """,
                inputs=[], outputs=[]
            )
            gr.Button("Exit").click(
                None,
                js="""
                () => {
                    document.body.innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100vh;font-size:1.5rem;">✅ Thank you! We will get back to you shortly.</div>';
                }
                """,
                inputs=[], outputs=[]
            )
            gr.HTML("<div id='camera-error'></div>")

    submit_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[one_time_id, email_input, interview_ui],
    )

if __name__ == "__main__":
    interview.launch(server_name="0.0.0.0", server_port=7860)
