import json
from pathlib import Path
import numpy as np
from PIL import Image
from dongfeng_registration.colmap_reconstruction import build_static_valid_masks


def test_colmap_mask_matches_symlink_target_and_original_image_size(tmp_path: Path):
    source=tmp_path/'raw/000000_00.jpg';source.parent.mkdir()
    Image.new('RGB',(8,4)).save(source)
    image_root=tmp_path/'images';(image_root/'camera_a').mkdir(parents=True)
    (image_root/'camera_a/1764650510701124.jpg').symlink_to(source)
    camera=tmp_path/'semantic/camera_a'
    (camera/'metadata').mkdir(parents=True);(camera/'class_masks').mkdir()
    labels=camera/'class_masks/1764650510701124000.png'
    Image.fromarray(np.array([[0,1]],dtype=np.uint8)).save(labels)
    (camera/'metadata/1764650510701124000.json').write_text(json.dumps({
        'source':{'path':str(source)},'class_mask_path':str(labels),
        'class_id_to_name':{'0':'road','1':'car'},
    }))
    out=tmp_path/'masks'
    result=build_static_valid_masks(semantic_root=tmp_path/'semantic',output_root=out,
        dynamic_classes={'car'},required_relative_names={'camera_a/1764650510701124.jpg'},image_root=image_root)
    mask=np.asarray(Image.open(out/'camera_a/1764650510701124.png'))
    assert mask.shape==(4,8)
    assert (mask[:,:4]==255).all() and (mask[:,4:]==0).all()
    assert result=={'masks':1,'dynamic_pixels':16}
    assert not (out/'camera_a/000000_00.png').exists()
