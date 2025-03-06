import spatialdata
import sopa
import novae

DEVICE = "cuda"
NUM_WORKERS = 12
PATIENCE = 10

LOAD_PATH = "/nfs/research/uhlmann/callum/data/ovarian/xenium/AU16/output-XETG00055__0031220__AU16-OVR-0-FO-1-S10__20240606__105527_baysor.zarr"
SAVE_PATH = "/nfs/research/uhlmann/callum/data/ovarian/xenium/AU16/output-XETG00055__0031220__AU16-OVR-0-FO-1-S10__20240606__105527_baysor_with_centroid_embeddings.zarr"



def main():
    sdata = spatialdata.read_zarr(
        LOAD_PATH
    )

    sopa.patches.compute_embeddings(
        sdata, 
        # model="resnet50", 
        model="random_noise", 
        # patch_width=56, 
        patch_width=256,
        # patch_overlap=28,
        batch_size=512,
        # patch_width=1024,
        level=1,
        image_key="morphology_focus",
        inference_channels = [0, 1, 2],
        device=DEVICE,
    )

    sopa.patches.centroid_image_embedding(
        sdata, 
        source_image_key="morphology_focus", 
        embedding_image_key="resnet50_embeddings",
    )

    sdata.write(SAVE_PATH)

    # sdata = spatialdata.read_zarr(
    #     SAVE_PATH
    # )


    adatas = [sdata.tables["table"]]

    novae.utils.spatial_neighbors(adatas, radius=15)
    model = novae.Novae(
        adatas, 
        num_prototypes=10, 
        feature_modality="image", 
        embedding_size=adatas[0].obsm["centroid_embeddings"].shape[1],        
    )

    model.fit(
        max_epochs=100, 
        accelerator=DEVICE,
        num_workers=NUM_WORKERS,
        patience=PATIENCE,
        )
    model.compute_representations(
        accelerator=DEVICE,
        num_workers=NUM_WORKERS,
        )

    model.save_pretrained("scripts/image_train_random")
    adatas[0].write_h5ad("scripts/image_train_random.h5ad")
    # model.save_pretrained("scripts/image_train")
    # adatas[0].write_h5ad("scripts/image_train.h5ad")

if __name__ == "__main__":
    main()