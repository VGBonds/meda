

def make_prompt_with_image(system_message, user_prompt, image):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},  # X-ray
                {"type": "image", "image": image}
            ]
        }
    ]
    return messages


def make_prompt_without_image(system_message, user_prompt):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},  # X-ray
                {"type": "image"}
            ]
        }
    ]
    return messages