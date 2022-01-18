FROM pytorch/pytorch

# Install dependencies
RUN python3 -m pip install timm
RUN apt-get update -y
RUN apt-get install python3-opencv -y
RUN python3 -m pip install opencv-python

# Execute the solver script
COPY ./solve.py /solve.py
CMD ["python3", "/solve.py", "/input", "/output"]