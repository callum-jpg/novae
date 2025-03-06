import novae
import sopa
import pytest
import numpy as np


def test_train_image():
    sdata = sopa.io.toy_dataset(length=2048, as_output=True, genes=10)

    sopa.patches.compute_embeddings(sdata, model="dummy", patch_width=256, image_key="image")

    sopa.patches.centroid_image_embedding(sdata, source_image_key="image", embedding_image_key="dummy_embeddings")

    adatas = [sdata.tables["table"]]

    novae.utils.spatial_neighbors(adatas)
    model = novae.Novae(
        adatas, 
        num_prototypes=10, 
        feature_modality="image", 
        embedding_size=adatas[0].obsm["centroid_embeddings"].shape[1]
    )

    with pytest.raises(AssertionError):  # should raise an error because the model has not been trained
        model.compute_representations()

    model.fit(max_epochs=3)
    model.compute_representations()
    model.compute_representations(num_workers=2)

    # obs_key = model.assign_domains(n_domains=2)
    obs_key = model.assign_domains(level=2)

    model.batch_effect_correction()

    adatas[0].obs.iloc[0][obs_key] = np.nan

    novae.monitor.mean_fide_score(adatas, obs_key=obs_key)
    novae.monitor.jensen_shannon_divergence(adatas, obs_key=obs_key)

    adatas[0].write_h5ad("tests/test.h5ad")  # ensures the output can be saved

    model.compute_representations(adatas, zero_shot=True)

    with pytest.raises(AssertionError):
        model.fine_tune(adatas, max_epochs=1)

    model.mode.pretrained = True

    model.fine_tune(adatas, max_epochs=1)