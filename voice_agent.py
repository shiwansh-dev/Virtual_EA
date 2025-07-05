from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from prompts import AGENT_INSTRUCTION, SESSION_INSTRUCTION


# ✅ Import your tools
from tools import get_weather, search_web, send_email

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=AGENT_INSTRUCTION,
            tools=[  # ✅ Include tools here
                get_weather,
                search_web,
                send_email,
            ]
        )


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # Create agent session using Deepgram STT
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-2",  # Use Deepgram’s best model
            language="hi",
        ),
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.7,
        ),
        tts=cartesia.TTS(
            model="sonic-2",
            voice="56e35e2d-6eb6-4226-ab8b-9776515a7094",
            language="hi",
        ),
        vad=silero.VAD.load(),
        turn_detection="stt",  # Use STT-based turn detection
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Greet the user when they join
    await session.generate_reply(
        instructions=SESSION_INSTRUCTION,
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
