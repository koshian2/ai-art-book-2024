mkdir -p /workdir/data/zunko02
cp /workdir/data/zunko_orig/*.png /workdir/data/zunko02/

python finetune/tag_images_by_wd14_tagger.py --onnx \
  --repo_id SmilingWolf/wd-swinv2-tagger-v3 \
  --batch_size 4  \
  --recursive \
  --use_rating_tags_as_last_tag \
  --character_tags_first \
  --always_first_tags "zunko,tohoku_zunko"  /workdir/data/zunko02