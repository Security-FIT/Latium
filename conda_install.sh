conda install python=3.11

pip install -r requirements.txt

case @1 in
    "amd")
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
        ;;

    *)
        pip install torch torchvision torchaudio 
        ;;
esac