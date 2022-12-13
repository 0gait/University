FROM openjdk:11-jdk-slim

LABEL build="docker build -t icl:tools ."

# Update & Install
RUN apt update && \
apt -y full-upgrade && \
apt -y install wget zip nano && \
rm -r /var/lib/apt/lists/* && \
#
# Install javacc
wget -O /opt/javacc.zip "https://github.com/javacc/javacc/archive/javacc-7.0.10.zip" && \
unzip -d /opt /opt/javacc.zip && rm /opt/javacc.zip && \
rm /opt/javacc-javacc-7.0.10/scripts/*.bat && \
chmod +x /opt/javacc-javacc-7.0.10/scripts/* && \
mkdir -p /opt/javacc-javacc-7.0.10/target && \
wget -O /opt/javacc-javacc-7.0.10/target/javacc.jar "https://repo1.maven.org/maven2/net/java/dev/javacc/javacc/7.0.10/javacc-7.0.10.jar" && \
#
# Install jasmin
wget -O /opt/jasmin.zip "https://sourceforge.net/projects/jasmin/files/latest/download" && \
unzip -d /opt /opt/jasmin.zip && rm /opt/jasmin.zip

COPY jasmin /bin/jasmin
RUN chmod +x /bin/jasmin

ENV PATH=$PATH:/opt/javacc-javacc-7.0.10/scripts/

CMD ["/bin/bash"]