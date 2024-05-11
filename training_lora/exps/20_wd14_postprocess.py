# 20_run_wd14.shを実行したあとに使う

import glob

def main():
    caption_files = sorted(glob.glob("../data/zunko02/*.txt"))
    append_tags = ["tohoku_zunko", "zunko"]
    remove_tags = ["touhoku_zunko"] # 表記揺れ
    for caption_file in caption_files:
        with open(caption_file, encoding="utf-8") as f:
            caption = f.read().strip()
        tags = caption.split(", ")

        for append_tag in append_tags:
            if append_tag not in tags:
                tags.insert(0, append_tag)
        
        for remove_tag in remove_tags:
            if remove_tag in tags:
                tags.remove(remove_tag)

        with open(caption_file, "w", encoding="utf-8") as fp:
            fp.write(", ".join(tags))

if __name__ == "__main__":
    main()