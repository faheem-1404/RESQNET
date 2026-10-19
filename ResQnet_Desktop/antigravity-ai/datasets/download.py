import kagglehub

print("Downloading datasets...")

flood = kagglehub.dataset_download(
    "saurabhshahane/roadway-flooding-image-dataset"
)

drone = kagglehub.dataset_download(
    "banuprasadb/visdrone-dataset"
)

weather = kagglehub.dataset_download(
    "nelgiriyewithana/global-weather-repository"
)

crowd = kagglehub.dataset_download(
    "tthien/shanghaitech-with-people-density-map"
)

print("Flood Dataset:", flood)
print("Drone Dataset:", drone)
print("Weather Dataset:", weather)
print("Crowd Dataset:", crowd)