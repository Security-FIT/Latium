conda install python=3.11

pip install -r requirements.txt

case @1 in
    "amd")
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
        ;;
    "nvidia")
        pip install torch torchvision torchaudio
        ;;
    "cpu")
        pip install torch torchvision torchaudio
        ;;
    *)
        echo "Please select one of the options: [amd, nvidia, cpu]"
        ;;
esac