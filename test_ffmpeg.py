import subprocess

# Run ffmpeg with '-codecs' option to list all supported codecs
result = subprocess.run(["ffmpeg", "-codecs"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Check if 'libx264' appears in the output
if "libx264" in result.stdout:
    print("libx264 is installed and supported.")
else:
    print("libx264 is NOT installed or supported.")
