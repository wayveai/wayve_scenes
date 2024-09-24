declare -a LINKS=(
    "https://drive.google.com/file/d/1nrpBYGhJZwPtoIwAef5tMKJz69fJFg18/view?usp=drive_link" 
    "https://drive.google.com/file/d/1Y83UVZTXHbFZV-jsxGuoU-X-DbDCBtnP/view?usp=drive_link" 
    "https://drive.google.com/file/d/1oNLxZOsZa4-R5wNSo3HvRhJQTdbto0lj/view?usp=drive_link" 
    "https://drive.google.com/file/d/1Q2QvQFyUrUcxh_ttxxkT3Q2Mf2XgnZC6/view?usp=drive_link" 
    "https://drive.google.com/file/d/1kXbb2EnhyYba401W7qBfHS0No5W7h6aR/view?usp=drive_link" 
    "https://drive.google.com/file/d/1xtRbeFjDLWuSxcHXfmkuSnjOzfm5rMOe/view?usp=drive_link" 
    "https://drive.google.com/file/d/181smXMRGSVGDlH1DcqODQpmiigVAnfx4/view?usp=drive_link" 
    "https://drive.google.com/file/d/1Sd9vurODCf5fyGkWdvXwX7r_yhi13O_M/view?usp=drive_link" 
    "https://drive.google.com/file/d/1uRiHfhW4zFK4QzH8rZg7hPemiuUE_Cad/view?usp=drive_link" 
    "https://drive.google.com/file/d/1F9fmmafeKVcrBr3MXhTguPvi_ucL7HDX/view?usp=drive_link" 
    "https://drive.google.com/file/d/19MwIyNh3cvFjg4DEmdmLbC0_WHPboZDn/view?usp=drive_link" 
    "https://drive.google.com/file/d/13xFsHuotbC8rGgpg43aPNkLhJ1f_bPqL/view?usp=drive_link" 
    "https://drive.google.com/file/d/1JPX20KLT6W1h69mzT2kMYga2iLI2BJlh/view?usp=drive_link" 
    "https://drive.google.com/file/d/1698IrtXxDA-0zPnc27_FsbpNrsZJ2Stb/view?usp=drive_link" 
    "https://drive.google.com/file/d/1GIpdCH9xALql651AJNjfd5fDHpBu0tUL/view?usp=drive_link" 
    "https://drive.google.com/file/d/1ZRk4GoGFwS2a0xalNAmAqfKIPO5j0i_7/view?usp=drive_link" 
    "https://drive.google.com/file/d/1aPxfl6LNshLhrWUR9Toj5zI9qNSxO9G8/view?usp=drive_link" 
    "https://drive.google.com/file/d/1Ch3eKVuwcadOQYPwSm7_sZgg9AKvXyny/view?usp=drive_link" 
    "https://drive.google.com/file/d/1lqI2UTQatTl6GzciP0hq5BuoH2yUdYJ8/view?usp=drive_link" 
    "https://drive.google.com/file/d/1zAr8rs5BA_YUQifIo5z4zf79VPgCff6e/view?usp=drive_link" 
    "https://drive.google.com/file/d/19xuEoZB0vAEny6AHGDfwAgc_YCRXLfFE/view?usp=drive_link" 
    "https://drive.google.com/file/d/1Uf5apT5z8oNLHAx9_Q7e3-lrJFKzdKL0/view?usp=drive_link" 
    "https://drive.google.com/file/d/1HnvZk21IacrnJDVFinptshJRPyFYNJpr/view?usp=drive_link" 
    "https://drive.google.com/file/d/1dwcYuvs13ftDr6zql2ZJd0b5Fodh7F8I/view?usp=drive_link" 
    "https://drive.google.com/file/d/1VkoqJK-435ZbSYTxjo0YlGTeXOOGky3o/view?usp=drive_link" 
    "https://drive.google.com/file/d/1T1jEqyPYqh5KgrfHLyCtzBZHULfjB3HI/view?usp=drive_link" 
    "https://drive.google.com/file/d/1XppE_OsSEjyHXY-4XCjoINKpsRG-3M4D/view?usp=drive_link" 
    "https://drive.google.com/file/d/1esiwteE-lU4mCkJKOxqs2vmYO1p9-6Lj/view?usp=drive_link" 
    "https://drive.google.com/file/d/1voYovE77GUy9WezecGtc2R4BSLtaNHsF/view?usp=drive_link" 
    "https://drive.google.com/file/d/15_YJoPleztQ0_jJG78xAJ2AgDodNK7if/view?usp=drive_link" 
    "https://drive.google.com/file/d/1fGvr8qlbPCl7fe63z7OVvHSzw5Ry_Nx_/view?usp=drive_link" 
    "https://drive.google.com/file/d/1aAbwd8huQCXsjEqX7gZQSQY0kKgHRErz/view?usp=drive_link" 
    "https://drive.google.com/file/d/1CjXRunQuCbDUtxLojYEhv6YqjOiwq3oH/view?usp=drive_link" 
    "https://drive.google.com/file/d/10jMElzQbp6DtG9m-F9EO4o_bpqWAa74L/view?usp=drive_link" 
    "https://drive.google.com/file/d/1UYxJpHWyVi57qeQrZ97JXX51azJH_0rm/view?usp=drive_link" 
    "https://drive.google.com/file/d/1q4-6yjaBXsGwh2sIso2jpm9pGrrfZJHj/view?usp=drive_link" 
    "https://drive.google.com/file/d/1VpMJsGfFGkAhQLKGoprflize4oSrAvbB/view?usp=drive_link" 
    "https://drive.google.com/file/d/13-1yOXrrmajdBLIkzahehAmcvca-y09A/view?usp=drive_link" 
    "https://drive.google.com/file/d/1D-FuyjpLjZAo92JpqcmjbW_PjaJPa3T6/view?usp=drive_link" 
    "https://drive.google.com/file/d/1wEcsQg5Y8G-P8vWbVB6gqYwz6ULnKjcb/view?usp=drive_link" 
    "https://drive.google.com/file/d/1CQ1ms9YItohdLfH-PdH9f7hf64Ki0rRP/view?usp=drive_link" 
    "https://drive.google.com/file/d/1Gdcb6qom5-ep7zoP1udrP4Hut-89NdDm/view?usp=drive_link" 
    "https://drive.google.com/file/d/1Q-3ABx73PbDGNarC6Eca1iuswzkpQIAE/view?usp=drive_link" 
    "https://drive.google.com/file/d/11hZsJ1SHL3aLZgAILK0mPlYsT3Ib85gk/view?usp=drive_link" 
    "https://drive.google.com/file/d/1FteDwKF-rnO1-q-SL47Cb1j-1WyM8U-m/view?usp=drive_link" 
    "https://drive.google.com/file/d/1x4rFMJQOcWTi1HDPfQxC8x_pOULuezT6/view?usp=drive_link" 
    "https://drive.google.com/file/d/1ClH33s2X1kgEezA4hJ5vJoZ14xVvGfL9/view?usp=drive_link" 
    "https://drive.google.com/file/d/1kmt7TculhlXHSpjQocj5YaJJu5ANcNej/view?usp=drive_link" 
    "https://drive.google.com/file/d/1RhJDQpZpLv9gNSuibsipyfKDb6vZUd59/view?usp=drive_link" 
    "https://drive.google.com/file/d/1N0c6x7k6NLr3QkhdfVZeSgVbUde2Od0g/view?usp=drive_link" 
    "https://drive.google.com/file/d/1A0sL_ogC3CNcOmJqOVUxU58Gzd0jWx-B/view?usp=drive_link" 
    "https://drive.google.com/file/d/1OFH3ITo8kgTK5U1XD96_X9NaQ_hQpQQL/view?usp=drive_link" 
    "https://drive.google.com/file/d/1yfbwaDALQp3jcDyu7yGD_BR7gKa-pc_d/view?usp=drive_link" 
    "https://drive.google.com/file/d/1LM0x_Ci420kPlRvvWvME0OGYTl1-RHpQ/view?usp=drive_link" 
    "https://drive.google.com/file/d/1EGLwwzO86dBoc3sthoDJ7S6n6nCfyl7Q/view?usp=drive_link" 
    "https://drive.google.com/file/d/1oHcdPfTD9U0VBQ6egw8CHzVno8m7VVH_/view?usp=drive_link" 
    "https://drive.google.com/file/d/1irE3eTk4Jjkj0BYqvm-DMGiZfZIpBMqs/view?usp=drive_link" 
    "https://drive.google.com/file/d/1n-QJA8K47u8XImdLSPnC40nWq7CAA_rI/view?usp=drive_link" 
    "https://drive.google.com/file/d/1CeI6dp5xuzrdkM3z2ygbobnWhFoLCi1Z/view?usp=drive_link" 
    "https://drive.google.com/file/d/1MjYi6Wudh_ZbCuUQUCoHLL-azPio2V-7/view?usp=drive_link" 
    "https://drive.google.com/file/d/1jKPWpyIea2VdtWYC6ZMfwapfw_fR055O/view?usp=drive_link" 
    "https://drive.google.com/file/d/1LOjkx_WDI6vTGJUHft4ZPiby4-1Ulj3C/view?usp=drive_link" 
    "https://drive.google.com/file/d/1z9H49KeRyjVFTavN1MyWshxH4YqzJOuf/view?usp=drive_link" 
    "https://drive.google.com/file/d/1TNRonwAkcXYCR-t5ViLNXz5gHwLDtJPa/view?usp=drive_link" 
    "https://drive.google.com/file/d/1zEbnZ0lXKYSkDwvQGhWr_5aMyFHKKn8k/view?usp=drive_link" 
    "https://drive.google.com/file/d/1w_6tOhMJ_NQOPveIoCdUDEC9gciVGJ_V/view?usp=drive_link" 
    "https://drive.google.com/file/d/1RJCnHDAE0LA1cd5gQ2sZ5VKs4A2CCP-e/view?usp=drive_link" 
    "https://drive.google.com/file/d/12XcHja6ucOQ8SALLYCetAjD2Ny91TsAM/view?usp=drive_link" 
    "https://drive.google.com/file/d/1M7_C4sYZO660_0FB1yM9kS4sZFRgEKL0/view?usp=drive_link" 
    "https://drive.google.com/file/d/1UqXyfRTz-RNMVD0jZXHWVF3WLeqSani2/view?usp=drive_link" 
    "https://drive.google.com/file/d/1gXLVUi_m6B1lL1_iKduHy2LhchsqH581/view?usp=drive_link" 
    "https://drive.google.com/file/d/1ltH0aML12BzPPavaPskoYxt-8tVsWdsJ/view?usp=drive_link"
    "https://drive.google.com/file/d/1ZTVdIujXKBRXTY5Vtx1zESo7rbvtNQPM/view?usp=drive_link" 
    "https://drive.google.com/file/d/1g-1nGXSRcD859Gicog_7x4brkZZU8Bur/view?usp=drive_link" 
    "https://drive.google.com/file/d/1ierf4ayQVO7fsMoCptW676hng3yhQiqv/view?usp=drive_link" 
    "https://drive.google.com/file/d/1SI5tqhzulmxifJecglJ9VJ8shccIJbxj/view?usp=drive_link" 
    "https://drive.google.com/file/d/17zvHHAmmSfSpoAMVxpnLEeA2ZRWdXOvZ/view?usp=drive_link"
    "https://drive.google.com/file/d/153PifGttbt8NMBclxzgWZ90PXhvGxGDy/view?usp=drive_link" 
    "https://drive.google.com/file/d/1ExdFHr3Etx9Ow5ajU9EM8QWcDBwPM4v-/view?usp=drive_link" 
    "https://drive.google.com/file/d/1REN087iTsCzdcvtgekIuWcOOszourxT7/view?usp=drive_link" 
    "https://drive.google.com/file/d/1Kic8btU-FHvY5K_DBaK6WO49rVoc2iGc/view?usp=drive_link"
    "https://drive.google.com/file/d/1IM_wbk6_QWb5-HmIWSb1sd7oVCREYPI8/view?usp=drive_link" 
    "https://drive.google.com/file/d/1nDc3WG3rgSKGkgm8__4ymNMqLG5k2QZ4/view?usp=drive_link" 
    "https://drive.google.com/file/d/1hhcZpp07wslZZQnPlDzRo2q87BXA97lQ/view?usp=drive_link" 
    "https://drive.google.com/file/d/1jSBBU6NLz8otR0HmDaXFRiCAQKtE4ua3/view?usp=drive_link" 
    "https://drive.google.com/file/d/1xEPinKI_VDz80qkGiiv120SCnXsXXQ-C/view?usp=drive_link" 
    "https://drive.google.com/file/d/1ZrCQCKyPbKJGvDxvYQnkT8KHRBK5Hkau/view?usp=drive_link" 
    "https://drive.google.com/file/d/1PF9XT0HidkuxySxF5kh0TWKgksEueDv2/view?usp=drive_link"
    "https://drive.google.com/file/d/1PZZXqq0nLotHpoLawm6-I5cU_Cl-qVfk/view?usp=drive_link" 
    "https://drive.google.com/file/d/1E_hMXgfvJAfidrCM09AdsJmO5Chr8uuA/view?usp=drive_link" 
    "https://drive.google.com/file/d/1CGV4PfwHP8U1_slteYm7zDJpYqMviMkr/view?usp=drive_link" 
    "https://drive.google.com/file/d/1a2fxOgTaKCZF196sTR-g-cFvKOp4lM-j/view?usp=drive_link" 
    "https://drive.google.com/file/d/1wCT1zzj29KY5OjopLALAKUGrGHf0qg3P/view?usp=drive_link" 
    "https://drive.google.com/file/d/1HI8cfLvWzBSpfq3TdoSw4D7wDZlmM-Yf/view?usp=drive_link" 
    "https://drive.google.com/file/d/1FRe3Py12Z48Ej9O_4s5K0hU_00uYmqwF/view?usp=drive_link" 
    "https://drive.google.com/file/d/1NDxkSn3sX3B-XDo6NKJL4jkVSu7UdpLi/view?usp=drive_link" 
    "https://drive.google.com/file/d/1y5AhWaosqkyndkEiVlIkhiRSj98Jh7P4/view?usp=drive_link" 
    "https://drive.google.com/file/d/1l8Jv_GmMfOXOVRPvyrLUBGlYficjra59/view?usp=drive_link" 
    "https://drive.google.com/file/d/1nVHQwm3_kAc8j-4X0dc-hdzuup5zyll-/view?usp=drive_link" 
    "https://drive.google.com/file/d/1B-24zThlMBIiKk5BzovKdVpIE89rnhs_/view?usp=drive_link" 
    "https://drive.google.com/file/d/1dv7QpxaqY5-YPewi_75JCR8LJCM-GH1e/view?usp=drive_link" 
    "https://drive.google.com/file/d/1rf3rAVE6Zw1S23UBhL6yobP5flR93ew5/view?usp=drive_link"
)

# Check if the user provided an argument (directory path)
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <download_directory>"
    exit 1
fi

# Directory where files will be downloaded
DOWNLOAD_DIR=$1

# Add / at the end of the directory path if it doesn't exist
if [[ "$DOWNLOAD_DIR" != */ ]]; then
    DOWNLOAD_DIR="$DOWNLOAD_DIR/"
fi

# Ensure the directory exists, if not try to create it
if [ ! -d "$DOWNLOAD_DIR" ]; then
    mkdir -p "$DOWNLOAD_DIR"
fi


# Iterate through each link and download the file to the specified directory
for link in "${LINKS[@]}"
do

    # Extract the file ID from the URL using sed
    id=$(echo "$link" | sed -n 's|https://drive.google.com/file/d/\(.*\)/view?usp=drive_link|\1|p')

    # Construct the download URL
    down_url="https://drive.google.com/uc?id=$id"

    echo "Downloading from $down_url to $DOWNLOAD_DIR"
    gdown "$down_url" -O "$DOWNLOAD_DIR"
done

echo "All files have been downloaded to $DOWNLOAD_DIR"

