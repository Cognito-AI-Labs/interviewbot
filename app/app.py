import pathlib
import hmac
import hashlib
import base64
import time
import os
from typing import Tuple, Dict, Any
from dotenv import load_dotenv
import gradio as gr
from fastrtc import WebRTC
import logging
import validate_code
from gemini_handler import GeminiHandler
from sheets_util import mark_exit_before_start
# Load environment variables from .env file
load_dotenv()
# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Set current directory for file references
current_dir: pathlib.Path = pathlib.Path(__file__).parent

# Read CSS and header HTML for Gradio interface
css: str = (current_dir / "assets" / "style.css").read_text()
header: str = (current_dir / "assets" / "header.html").read_text()

def generate_turn_credentials(secret: str, ttl: int = 3600) -> Tuple[str, str]:
    """
    Generate TURN server credentials using a shared secret and time-to-live (ttl).
    Args:
        secret (str): The shared TURN secret.
        ttl (int, optional): Time-to-live in seconds for the credential. Defaults to 3600 seconds (1hr).
    Returns:
        Tuple[str, str]: A tuple containing the username and credential for TURN authentication.
    """
    username: str = str(int(time.time()) + ttl)
    key: bytes = bytes(secret, "utf-8")
    digest: bytes = hmac.new(key, username.encode("utf-8"), hashlib.sha1).digest()
    credential: str = base64.b64encode(digest).decode("utf-8")
    return username, credential

def get_rtc_configuration() -> Dict[str, Any]:
    """
    Returns the RTC (WebRTC) configuration dictionary for ICE servers.
    Returns:
        Dict[str, Any]: RTC configuration including STUN and TURN servers.
    """
    username, credential = generate_turn_credentials(os.environ["TURN_SECRET"])
    return {
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
            {"urls": "stun:stun2.l.google.com:19302"},
            {
                "urls": "turn:interview.cognitolabs.ai:3478",
                "username": username,
                "credential": credential,
            },
        ],
        "iceCandidatePoolSize": 10,
    }

def show_ui_if_validated(success: bool) -> gr.update:
    """
    Show or hide the interview UI based on validation success.
    Args:
        success (bool): Whether the validation was successful.
    Returns:
        gr.update: Gradio update object to control UI visibility.
    """
    return gr.update(visible=True) if success else gr.update(visible=False)

def handle_early_exit():
    mark_exit_before_start()

# Gradio Blocks UI definition
with gr.Blocks(
    css=css,
    title="Candidate Screening",
    js="""
    () => {

        // MutationObserver to handle interview start/stop and timer logic
        const observer = new MutationObserver(() => {
            // Find and bind timer to "Start Interview" button
            const startBtn = Array.from(document.querySelectorAll("div.icon-with-text"))
                .find(div => div.textContent.trim().toLowerCase().includes("start interview"));
            if (startBtn && !startBtn.dataset.timerBound) {
                startBtn.dataset.timerBound = "true";
                startBtn.addEventListener("click", () => {
                    console.log("[TIMER] Interview started");
                    setTimeout(() => {
                        console.log("[TIMER] Time expired");
                        endInterviewAndShowThankYou();
                    }, 920000);  // 15 min 20 sec
                });
            }
            // Find and bind handler to "Stop Interview" button
            const stopBtn = Array.from(document.querySelectorAll("div.icon-with-text"))
                .find(div => div.textContent.trim().toLowerCase().includes("stop interview"));
            if (stopBtn && !stopBtn.dataset.stopBound) {
                stopBtn.dataset.stopBound = "true";
                stopBtn.addEventListener("click", () => {
                    console.log("[STOP] Interview manually stopped");
                    endInterviewAndShowThankYou();
                });
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });

        // Function to end interview and show thank you message
        function endInterviewAndShowThankYou() {
            const interviewUI = document.getElementById("interview-ui");
            if (interviewUI) interviewUI.style.display = "none";
            document.body.innerHTML = `
                <div style="display:flex;justify-content:center;align-items:center;height:100vh;text-align:center;font-size:1.8rem;font-weight:bold;">
                    ✅ Thank you! We will get back to you shortly.
                </div>
            `;
        }
    }
    """
) as interview:
    gr.HTML(header)
    # Candidate input section
    with gr.Column():
        one_time_id: gr.Textbox = gr.Textbox(label="Enter One-Time Interview Code", placeholder="12 digit code")
        name_input: gr.Textbox = gr.Textbox(label="Enter Your Name", placeholder="Full Name")
        email_input: gr.Textbox = gr.Textbox(label="Enter Your Email", placeholder="name@example.com")
        submit_btn: gr.Button = gr.Button("Next")
    # Camera error popup (hidden by default)
    gr.HTML("""
        <div id="camera-popup" style="
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #ffdddd;
            color: #b30000;
            padding: 12px 16px;
            border: 1px solid #b30000;
            border-radius: 8px;
            font-weight: bold;
            display: none;
            z-index: 9999;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.2);
        "></div>
    """)
    # Main interview UI (hidden until validation)
    with gr.Column(elem_id="camera-container", visible=False) as interview_ui:
        with gr.Column(scale=1):
            # Webcam feed area
            gr.HTML("""
                <div style="display: flex; justify-content: center;">
                    <video id="webcam-feed" autoplay muted playsinline
                        style="width: 100%; max-width: 420px; max-height: 280px; object-fit: cover; border-radius: 1px; border: 1px solid #ccc;">
                    </video>
                    <div id="camera-error" style="text-align: center;"></div>
                </div>
            """)
            # Enable Camera button (initially disabled)
            gr.Button("Enable Camera", elem_id="enable-camera-btn", interactive=False).click(
                None,
                js="""
                () => {
                    // Enable webcam and show audio section
                    const video = document.getElementById('webcam-feed');
                    const popup = document.getElementById('camera-popup');
                    const enableBtn = document.getElementById('enable-camera-btn');
                    const log = (msg) => {
                        const area = document.getElementById('log-area');
                        if (area) area.innerText += msg + '\\n';
                    };
                    if (popup) popup.style.display = 'none';
                    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
                        .then(stream => {
                            video.srcObject = stream;
                            if (enableBtn) enableBtn.style.display = 'none';
        
                            const audioSection = document.getElementById('audio-section');
                            if (audioSection) {
                                audioSection.style.display = 'block';
                                audioSection.scrollIntoView({ behavior: "smooth" });
                            } else {
                                log("❌ audio-section not found.");
                            }
                        })
                        .catch(error => {
                            console.error('Camera access error:', error);
                            log("❌ Please enable camera: " + error.name);
                            if (popup) {
                                if (error.name === 'NotAllowedError') {
                                    popup.innerText = '❌ Please give camera access in browser settings.';
                                } else {
                                    popup.innerText = 'Camera error: ' + error.name;
                                }

                                popup.style.display = 'block';
                                setTimeout(() => popup.style.display = 'none', 10000);
                            }
                        });
                }
                """,
                inputs=[],
                outputs=[]
            )

        with gr.Column():
            # Audio streaming section (WebRTC)
            with gr.Group(elem_id="audio-section", elem_classes=["no-scroll-audio"], visible=True) as audio_group:
                rtc_configuration: Dict[str, Any] = get_rtc_configuration()
                webrtc: WebRTC = WebRTC(
                    label="🎙️ Interview Audio Stream",
                    modality="audio",
                    mode="send-receive",
                    rtc_configuration=rtc_configuration,
                    button_labels={"start": "Start Interview", "stop": "Stop Interview"},
                )
                # Stream audio to GeminiHandler for processing/interview
                webrtc.stream(
                    GeminiHandler(),
                    inputs=[webrtc],
                    outputs=[webrtc],
                    time_limit=900,  # 15 minutes
                    concurrency_limit=2
                )

            # Exit button to end interview and show thank you message
            gr.Button("Exit", elem_classes=["shared-purple-btn"]).click(
                fn=handle_early_exit,
                inputs=[],
                outputs=[]
            ).then(
                None,
                js="""() => {
                    document.body.innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100vh;font-size:1.5rem;">✅ Thank you! We will get back to you shortly.</div>';
                }""",
                inputs=[],
                outputs=[]
            )


        # success_flag is a Gradio State object used to keep track of whether the code validation was successful.
        # It is set by the validate_code function and then used to determine if the interview UI should be shown.
        success_flag: gr.State = gr.State()
        # Submit button triggers code validation and UI reveal
        submit_btn.click(
            validate_code.validate_code,
            inputs=[one_time_id, name_input, email_input],
            outputs=[one_time_id, name_input, email_input, submit_btn, success_flag]
        ).then(
            fn=show_ui_if_validated,
            inputs=success_flag,
            outputs=interview_ui
        ).then(
            None,
            js="""() => {
                // Enable the camera button after validation
                const enableBtn = document.getElementById('enable-camera-btn');
                if (enableBtn) {
                    enableBtn.disabled = false;
                    enableBtn.style.opacity = '1';
                }
            }""",
            inputs=[],
            outputs=[]
        )

if __name__ == "__main__":
    # Launch the Gradio app
    interview.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path=str(current_dir / "assets" / "cognito_icon.png")
    )
