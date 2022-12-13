FROM henriquej0904/icl:tools AS build

COPY ./src /app/src

WORKDIR /app/src

# Compile
RUN mkdir ../bin && \
javacc Parser.jj && \
find | grep -e ".java" > ../files.txt && \
javac -d ../bin @../files.txt && \
rm ../files.txt

# Create jar
WORKDIR /app/bin

RUN find | grep -e ".class" > ../class-files.txt && \
jar --create --file ../MathExpression.jar -e MathExpression @../class-files.txt && \
rm ../class-files.txt

# ------------------------------------------------------------

FROM henriquej0904/icl:tools

WORKDIR /app

# Copy jar and build files
COPY --from=build /app/MathExpression.jar /app/MathExpression.jar