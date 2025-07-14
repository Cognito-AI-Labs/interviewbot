import asyncio
import pathlib
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
    GenerationConfig,
    AutomaticActivityDetection,
    EndSensitivity,
    ActivityHandling,
    ProactivityConfig,
    TurnCoverage,
    StartSensitivity,
)
import os
import io
import json
from dotenv import load_dotenv
import wave
from fastrtc import AsyncStreamHandler
from sheets_util import get_google_sheets_service, fetch_prompt_from_sheet
import numpy as np
import logging
import validate_code
from datetime import datetime, timezone
import boto3
load_dotenv()
# logger config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
current_dir = pathlib.Path(__file__).parent

# spreadsheet details and s3 bucket name
SPREADSHEET_ID = os.environ["SPREADSHEET_ID"]  
s3 = boto3.client("s3") 
bucket_name = os.environ["S3_BUCKET"]

class GeminiHandler(AsyncStreamHandler):
    """Handler for the Gemini API"""
    def __init__(
        self,
        expected_layout: Literal["mono"] = "mono",
        output_sample_rate: int = 24000,
        output_frame_size: int = 480,
        input_sample_rate: int = 16000,
        session_handle: Optional[str] = None,
        is_resumable: bool = False,
        is_goaway_recieved: bool = False, 
    ) -> None:
        super().__init__(
            expected_layout,
            output_sample_rate,
            output_frame_size,
            input_sample_rate=input_sample_rate,
        )
        self.session_handle = session_handle
        self.is_resumable: asyncio.Event = asyncio.Event()
        self.is_goaway_recieved: asyncio.Event = asyncio.Event()
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.quit: asyncio.Event = asyncio.Event()
        self.transcripts = []
        self.current_gemini_utterance = ""
        self.current_user_utterance = ""
        self.candidate_audio = bytearray()
        self.start_time = None
        self.end_time = None
        self.status = "unknown"
        self.status_message = ""
        if is_resumable:
            self.is_resumable.set()
        if is_goaway_recieved:
            self.is_goaway_recieved.set()

    def copy(self) -> "GeminiHandler":
        """Required implementation of the copy method for AsyncStreamHandler"""
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
        """Receive audio from the user and put it in the input queue"""
        _, array = frame
        array = array.squeeze()
        audio_message = array.tobytes()
        self.input_queue.put_nowait(audio_message)
        self.candidate_audio.extend(audio_message)

    async def emit(self) -> tuple[int, np.ndarray]:
        """Required implementation of the emit method for AsyncStreamHandler"""
        while True:
            audio_bytes = await self.output_queue.get()
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            return (self.output_sample_rate, audio_array)

    async def initialize_genai_session(self) -> None:
        """Initialize the GeminiHandler"""
        logger.info(f"Connecting to session..")
        SYSTEM_INSTRUCTION = fetch_prompt_from_sheet()
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
            session_resumption=SessionResumptionConfig(
                handle=self.session_handle,
            ),
            proactivity=ProactivityConfig(proactive_audio=True),
            realtime_input_config=RealtimeInputConfig(
                automatic_activity_detection=AutomaticActivityDetection(
                    disabled=False,
                    end_of_speech_sensitivity=EndSensitivity.END_SENSITIVITY_HIGH,
                    start_of_speech_sensitivity=StartSensitivity.START_SENSITIVITY_HIGH,
                    prefix_padding_ms=300,
                    silence_duration_ms=800,
                ),
                activity_handling=ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                turn_coverage=TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            ),
        )
        try:
            async with client.aio.live.connect(
                model=os.environ["LIVE_MODEL"], config=config
            ) as self.session:
                if self.session_handle is None:
                    await self.session.send_client_content(
                        turns=[
                            Content(
                            parts=[Part.from_text(text="Hi!")],
                            role="user",
                        )
                        ],
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
        """Optional asynchronous startup logic. Must be a coroutine (async def)."""
        self.start_time = datetime.now(timezone.utc)
        await self.initialize_genai_session()


    def shutdown(self) -> None:
        """Gracefully stop the stream and save transcript/audio."""
        logger.info("Shutting down")
        self.is_resumable.clear()
        self.quit.set()
        self.result = "interview_complete"
        self.finished = True
        
        # Ensure Sheets API client is initialized
        if not hasattr(self, "sheet") or self.sheet is None:
            sheet_service = get_google_sheets_service()
            self.sheet = sheet_service.spreadsheets() if sheet_service else None

        # Flush pending transcripts
        if self.current_user_utterance:
            self.transcripts.append({
                "speaker": "user",
                "text": self.current_user_utterance.strip()
            })
            self.current_user_utterance = ""

        if self.current_gemini_utterance:
            self.transcripts.append({
                "speaker": "gemini",
                "text": self.current_gemini_utterance.strip()
            })
            self.current_gemini_utterance = ""

        # Record end time and duration
        self.end_time = datetime.now(timezone.utc)
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.start_str = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.end_str = self.end_time.strftime("%Y-%m-%d %H:%M:%S")
        self.duration_str = f"{int(self.duration_seconds // 60)}m {int(self.duration_seconds % 60)}s"

        if not self.candidate_audio:
            logger.warning("No audio recorded.")
            self.status = "Failed"
            self.status_message = "No audio captured during session."
        elif not self.transcripts:
            logger.warning("No transcript generated.")
            self.status = "Failed"
            self.status_message = "No transcript generated."
        else:
            self.status = "Success"
            self.status_message = ""

        logger.info(f"[Shutdown] Saving audio and transcripts")
        self.save_transcripts_and_audio()

    def save_transcripts_and_audio(self) -> None:
        """Upload audio to S3 and update Google Sheet with transcript and audio link."""
        try:
            transcript_str = json.dumps(self.transcripts, ensure_ascii=False, indent=2)
            if not self.sheet:
                logger.error("Sheet API not initialized — cannot save to Google Sheet.")
                return
            # Prepare in-memory WAV to store the candidate recording
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.input_sample_rate)
                wf.writeframes(self.candidate_audio)
            buffer.seek(0)

            # Upload to S3
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"{validate_code.CURRENT_CODE}.wav"
            s3_key = f"interview_recordings/{date_str}/{filename}"
            s3.upload_fileobj(
                Fileobj=buffer,
                Bucket=bucket_name,
                Key=s3_key,
                ExtraArgs={"ContentType": "audio/wav"}
            )
            s3_url = f"https://{bucket_name}.s3.us-east-1.amazonaws.com/{s3_key}"
            logger.info(f"Audio uploaded to {s3_url}")
            sheet = self.sheet
            result = sheet.values().get(
                spreadsheetId=SPREADSHEET_ID,
                range="interview_tracker!A:Z"
            ).execute()
            rows = result.get("values", [])
            found_row = next((i for i, row in enumerate(rows) if row and row[0] == validate_code.CURRENT_CODE), -1)

            if found_row == -1:
                logger.error(f"Code {validate_code.CURRENT_CODE} not found in sheet.")
                self.status = "Failed"
                self.status_message = "Code not found in Google Sheet"
            update_range = f"interview_tracker!F{found_row + 1}:O{found_row + 1}"
            values = [[
                self.start_str,
                self.end_str,
                self.duration_str,
                transcript_str,
                s3_url,
                self.status,
                self.status_message
            ]]
            sheet.values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=update_range,
                valueInputOption="RAW",
                body={"values": values}
            ).execute()
            logger.info(f"Sheet updated for code {validate_code.CURRENT_CODE}")

        except Exception as e:
            logger.error(f"Failed to save transcripts or upload audio: {e}")
            self.status = "Failed"
            self.status_message = str(e)
            if self.sheet:
                try:
                    result = self.sheet.values().get(
                        spreadsheetId=SPREADSHEET_ID,
                        range="interview_tracker!A:Z"
                    ).execute()
                    rows = result.get("values", [])
                    found_row = next((i for i, row in enumerate(rows) if row and row[0] == validate_code.CURRENT_CODE), -1)
                    if found_row != -1:
                        update_range = f"interview_tracker!K{found_row + 1}:L{found_row + 1}"
                        self.sheet.values().update(
                            spreadsheetId=SPREADSHEET_ID,
                            range=update_range,
                            valueInputOption="RAW",
                            body={"values": [[self.status, self.status_message]]}
                        ).execute()
                except Exception as e2:
                    logger.error(f"Also failed to update status in sheet: {e2}")


    async def send_to_genai(self) -> AsyncGenerator[bytes, None]:
        """Helper method to stream input audio to the server. Used in start_stream."""
        while not self.quit.is_set():
            try:
                audio = await asyncio.wait_for(self.input_queue.get(), timeout=30)
                content = Blob(
                data=audio, mime_type=f"audio/pcm;rate={self.input_sample_rate}"
                )
                await self.session.send_realtime_input(audio=content)
            except Exception as e:
                logger.error(f"Unhandled error while sending audio: {e}")
                self.quit.set()
                break

    async def process_genai_response(self) -> AsyncGenerator[bytes, None]:
        """Helper method to stream input audio to the server. Used in start_stream.
        - Recieve audio from genAI and put it in the output queue
        - If new session id is recieved, save the session id as self.session_handle
        - If goAway is recieved, set the quit event, and intiliase a new session
        - If user interrupt is recieved, clear is output queue
        """
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
                            logger.info("Interrupted")
                            self.clear_queue()

                       # Handle input transcription (user)
                        if response.server_content.input_transcription:
                            text = response.server_content.input_transcription.text
                            if text:
                                if self.current_gemini_utterance.strip():
                                    self.transcripts.append({
                                        "speaker": "gemini",
                                        "text": self.current_gemini_utterance.strip()
                                    })
                                    self.current_gemini_utterance = ""
                                
                                self.current_user_utterance += text

                        # Handle output transcription (gemini)
                        if response.server_content.output_transcription:
                            text = response.server_content.output_transcription.text
                            if text:
                                if self.current_user_utterance.strip():
                                    self.transcripts.append({
                                        "speaker": "user",
                                        "text": self.current_user_utterance.strip()
                                    })
                                    self.current_user_utterance = ""
                                
                                self.current_gemini_utterance += text

                        # Fetching model output
                        if response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data.data:
                                    self.output_queue.put_nowait(part.inline_data.data)
            
                        # Session resumption if model sends go_away
                        if response.go_away:
                            time_left = response.go_away.time_left
                            logger.warning(
                                f"Received go away signal. Time left: {time_left}"
                            )
                            self.is_goaway_recieved.set()

            except Exception as e:
                logger.error(f"Unhandled error in receiving content: {e}")
                self.quit.set()
                break
