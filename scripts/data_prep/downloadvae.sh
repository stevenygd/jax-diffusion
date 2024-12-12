curr_dir=`pwd`

# Set the save directory
save_dir=`pwd`/data/vae
mkdir -p $save_dir

curl -L -o $save_dir/params.npy "https://drive.google.com/uc?export=download&id=1gec7PVqthCalDyJ-LJrVbBUyaKKgu95T&confirm=yes"
curl -L -o $save_dir/config.npy "https://drive.google.com/uc?export=download&id=1Rm1j4amMj6tF1w_ssa7N8TALu5qh37GV"

echo "Downloaded VAE parameters and config to $save_dir"

# Change back to the current directory
cd $curr_dir