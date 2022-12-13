#! /bin/bash
docker run --rm -it -v "$(pwd)/app/examples:/app/examples" -v "/etc/passwd:/etc/passwd:ro" -u $UID icl:level3