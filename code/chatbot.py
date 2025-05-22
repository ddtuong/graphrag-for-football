import chainlit as cl
from core import process_question

@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...

    # Send a response back to the user
    await cl.Message(
        content=f"Answer: {process_question(message.content)}",
    ).send()