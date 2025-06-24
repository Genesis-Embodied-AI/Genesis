import cv2
import base64
import openai
import os


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_PJ")

# 分析したい動画のパス
VIDEO_PATH = "data/videos/grasp_can_aluminium.mp4"

FRAMES_TO_EXTRACT = [0, 50, 100, 150]

system_prompt = """
You are a professional in manipulation and analysis of images.
You are tasked with analyzing a set of images extracted from a video.
Focus on information needed to manipulate the objects in the images.
"""
user_prompt = """
The following images are extracted from a video. Tell me what is happening in these images.
You should focus on the end-effector and the object being manipulated.
If the end-effector never touches the object, do not mention the object.
If the end-effector touches the object, carefully describe the interaction, such as grasping, pinching, and pushing.
Do not mention the shape or color of the object; simply refer to it as "the object".
If the end-effector is touching the object, explain the level of deformation using the following scale:
'none', 'slight', 'moderate', 'heavy', or 'extreme'.
Also, if the end-effector is grasping the object, describe whether the grasp seems stable or unstable.
Output only the analysis of the images. Do not include any other information. Only use alphabetical characters, spaces, commas, and periods.
Output in approximately 15 words.
"""

# --- 関数定義 ---

def extract_and_encode_frame(video_path: str, frame_nums: list[int]) -> list[str | None]:
    """
    OpenCVを使い、動画からフレームを抽出して直接Base64エンコードする。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルが開けません: {video_path}")
        return [None] * len(frame_nums)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if any(frame_num < 0 or frame_num >= total_frames for frame_num in frame_nums):
        print(f"エラー: 無効なフレーム番号です。0 から {total_frames - 1} の間で指定してください。")
        cap.release()
        return [None] * len(frame_nums)

    encoded_images = []
    for frame_num in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            print(f"エラー: フレーム {frame_num} の読み込みに失敗しました。")
            encoded_images.append(None)
            continue

        # フレームをメモリ上でPNG形式にエンコード
        success, encoded_image_binary = cv2.imencode('.png', frame)
        if not success:
            print(f"エラー: フレーム {frame_num} のエンコードに失敗しました。")
            encoded_images.append(None)
            continue

        # Base64文字列に変換
        base64_string = base64.b64encode(encoded_image_binary).decode('utf-8')
        encoded_images.append(base64_string)

    cap.release()
    return encoded_images 


def analyze_image_with_openai(base64_images: list[str | None]):
    """
    Base64エンコードされた画像をOpenAI APIに送信して分析する。
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY":
        print("エラー: OpenAI APIキーが設定されていません。")
        return None

    for base64_image in base64_images:
        if base64_image is None:
            raise ValueError("Base64エンコードされた画像が無効です。")

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="o3",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_images[0]}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_images[1]}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_images[2]}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_images[3]}"},
                        }
                    ]
                }
            ],
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"エラー: OpenAI APIの呼び出し中に問題が発生しました: {e}")
        return None


if __name__ == "__main__":
    print(f"{VIDEO_PATH} の {FRAMES_TO_EXTRACT} フレーム目を分析します...")

    base64_image_string = extract_and_encode_frame(VIDEO_PATH, FRAMES_TO_EXTRACT)

    if base64_image_string:
        analysis_result = analyze_image_with_openai(base64_image_string)
        
        if analysis_result:
            print("\n--- 分析結果 ---")
            print(analysis_result)
            print("-----------------")
    else:
        print("処理を中断しました。")