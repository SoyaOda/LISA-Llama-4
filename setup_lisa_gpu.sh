#!/bin/bash

# 使用方法を確認
if [ "$#" -lt 2 ]; then
    echo "使用方法: $0 <IPアドレス> <HuggingFaceトークン>"
    echo "例: $0 152.67.12.122 hf_************************"
    exit 1
fi

IP_ADDRESS=$1
HF_TOKEN=$2

echo "🚀 LISAセットアップを開始します: $IP_ADDRESS"

# リモートサーバーで実行するコマンドを定義
run_remote_cmd() {
    ssh -o StrictHostKeyChecking=no ubuntu@$IP_ADDRESS "$1"
}

# SSHキーを設定
echo "🔑 SSHキーを設定中..."
ssh-keyscan -H $IP_ADDRESS >> ~/.ssh/known_hosts

# 基本的な依存関係のインストール
echo "📦 基本的な依存関係をインストール中..."
run_remote_cmd "sudo apt-get update && sudo apt-get install -y git python3-pip python3-venv wget"

# リポジトリをクローン
echo "📥 リポジトリをクローン中..."
run_remote_cmd "rm -rf ~/LISA-Llama-4 && git clone https://github.com/yourusername/LISA-Llama-4.git ~/LISA-Llama-4"

# 仮想環境の作成
echo "🐍 Python仮想環境を作成中..."
run_remote_cmd "cd ~/LISA-Llama-4 && python3 -m venv lisa_env && source lisa_env/bin/activate && pip install --upgrade pip"

# 必要なパッケージのインストール
echo "📚 必要なパッケージをインストール中..."
run_remote_cmd "cd ~/LISA-Llama-4 && source lisa_env/bin/activate && pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118"
run_remote_cmd "cd ~/LISA-Llama-4 && source lisa_env/bin/activate && pip install -r requirements.txt"
run_remote_cmd "cd ~/LISA-Llama-4 && source lisa_env/bin/activate && pip install deepspeed==0.10.0 scikit-image torchvision accelerate bitsandbytes"

# Hugging Faceトークンの設定
echo "🔐 Hugging Faceトークンを設定中..."
run_remote_cmd "cd ~/LISA-Llama-4 && source lisa_env/bin/activate && huggingface-cli login --token $HF_TOKEN"

# 小さなテストデータセットのダウンロード
echo "🌍 テストデータセットをダウンロード中..."
run_remote_cmd "cd ~/LISA-Llama-4 && source lisa_env/bin/activate && pip install gdown && mkdir -p small_test_dataset && cd small_test_dataset && gdown --folder https://drive.google.com/drive/folders/your-folder-id"

# SAMモデルのダウンロード
echo "🧠 SAMモデルをダウンロード中..."
run_remote_cmd "cd ~/LISA-Llama-4 && source lisa_env/bin/activate && mkdir -p checkpoints && cd checkpoints && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# CUDAの設定
echo "🖥️ CUDAを設定中..."
run_remote_cmd 'echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64" >> ~/.bashrc'
run_remote_cmd 'echo "export PATH=\$PATH:/usr/local/cuda-11.7/bin" >> ~/.bashrc'
run_remote_cmd "source ~/.bashrc"

# 完了メッセージ
echo "✅ セットアップが完了しました！"
echo "サーバーにSSH接続するには: ssh ubuntu@$IP_ADDRESS"
echo "仮想環境をアクティベートするには: source lisa_env/bin/activate"
echo "モデルを学習するには次のコマンドを実行します:"
echo "deepspeed --master_port=24999 train_ds.py --version=\"meta-llama/Llama-3.2-11B-Vision-Instruct\" --dataset_dir='./small_test_dataset' --vision_pretrained=\"./checkpoints/sam_vit_h_4b8939.pth\" --vision-tower=\"meta-llama/Llama-3.2-11B-Vision-Instruct\" --dataset=\"sem_seg||refer_seg||vqa||reason_seg\" --sample_rates=\"9,3,3,1\" --exp_name=\"lisa-llama3-2-vision-test\" --conv_type=\"llama_3\" --precision=\"bf16\" --batch_size=1 --grad_accumulation_steps=1 --steps_per_epoch=2 --epochs=1" 