#! /bin/bash

mkdir -p app/jFiles

docker build -t icl:level3-debug --build-arg JAVA_DEBUG_OPTIONS="-g" . && \
docker run --rm -it -v "$(pwd)/app/examples:/app/examples" \
-v "$(pwd)/app/jFiles:/app/jFiles" \
-v "/etc/passwd:/etc/passwd:ro" -u $UID --network host -e DEBUG=1 icl:level3-debug