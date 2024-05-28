## 構成
* renders ：　カメラポーズを補間してレンダリングされた動画
    * street_360の元動画：https://360rtc.com/videos/sakaemachi002/

## 始め方
```console
git clone https://github.com/koshian2/nerfstudio -b v1.0.0-fix

cd nerfstudio
```

docker-compose.yamlのGPUIDを必要なら修正

```yaml
services:
  nerfstudio:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        CUDA_VERSION: 11.8.0
        CUDA_ARCHITECTURES: 86
        OS_VERSION: 22.04
    image: nerfstudio
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0'] # change if needed
              capabilities: [gpu]
    volumes:
      - ./workspace:/workspace/
      - ./.cache/:/home/user/.cache/
    ports:
      - "7007:7007"
      - "5000:5000"
    environment:
      - CUDA_VISIBLE_DEVICES=0 # change if needed
    shm_size: '12gb'      
    restart: "no"
    tty: true
    stdin_open: true
    ipc: host
```

Dockerイメージのビルド（30分程度必要）

```console
docker-compose build
```

コンテナの起動


```console
docker-compose run --service-ports nerfstudio
```


