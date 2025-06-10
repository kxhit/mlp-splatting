from train import train_scene_llff

if __name__ == "__main__":
    data_dir = "/home/xin/data/nerf/nerf_example_data/nerf_llff_data/fern"
    train_scene_llff(data_dir, num_epochs=10000, batch_size=4096*32)