import os
import subprocess
from huggingface_hub import login

def login_huggingface(token=None):
    """
    Hugging Faceへのログイン処理
    Args:
        token: Hugging Faceのトークン（Noneの場合は環境変数またはコマンドラインから取得）
    """
    if token is None:
        # 環境変数からトークンを取得
        token = os.environ.get("HUGGINGFACE_TOKEN")
    
    if token:
        # PythonのAPIを使用してログイン
        login(token=token)
        print("Hugging Faceにログインしました（Python API経由）")
    else:
        # コマンドラインでログイン
        try:
            subprocess.run(["huggingface-cli", "login"], check=True)
            print("Hugging Faceにログインしました（CLI経由）")
        except subprocess.CalledProcessError:
            print("Hugging Faceへのログインに失敗しました。トークンを確認してください。")
            return False
    return True

if __name__ == "__main__":
    # スクリプトとして実行された場合、ログインを試みる
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        login_huggingface(token)
    else:
        print("環境変数HUGGINGFACE_TOKENが設定されていません。コマンドラインログインを試みます。")
        login_huggingface() 