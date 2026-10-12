# Green node image.
#   docker build -t green .
#   docker run -it -v green-data:/data green wallet new
#   docker run -it -v green-data:/data green genesis --msg "..."          (or copy a genesis.json into the volume)
#   docker run -d -p 8546:8546 -v green-data:/data -e GREEN_WALLET_PASSWORD=... --name green green
# Port 8546 is the encrypted P2P protocol other nodes connect to; publish the API
# port too (-p 127.0.0.1:8545:8545) only if tools on the host need it. Admin
# endpoints (mining switch, peers, blob upload) only answer from inside the
# container:  docker exec green grn --data /data mine off
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY green ./green
RUN pip install --no-cache-dir . && useradd --create-home --uid 10001 green && mkdir /data && chown green /data

USER green
VOLUME /data
EXPOSE 8545 8546
ENTRYPOINT ["grn", "--data", "/data"]
CMD ["node", "--host", "0.0.0.0", "--mine"]
