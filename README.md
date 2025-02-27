## Overview
This is an Android-based application for **real-time human activity recognition** using IoT sensors (Thingy, RESpeck). It leverages **machine learning** to classify activities and monitor real-time data.

---

## Requirements
- **Android SDK:** API 34+ (Android 13 or higher)
- **Minimum SoC:** Snapdragon 625 or higher
- **Minimum Storage:** 1GB free space
- **Minimum RAM:** 512MB

## Installation Guide

### Step 1: Download the App
#### Method 1: Transfer APK from Computer
1. Download `{PDIoT_Group_X1.apk}` from the Learn page.
2. Locate the file (Windows: `C:\Users\YourUserName\Downloads`).
3. Connect your phone via a **USB cable**.
4. Set connection mode to **File Transfers** (MTP mode).
5. Copy the APK file to your phone’s storage.

#### Method 2: Download APK Directly on Phone
1. Open a browser on your phone.
2. Download the APK from the **GitHub repository**:
   [PDIoT Group X1 APK Release](https://github.com/Shu-Gu/PDIoT_X1_Public/releases/tag/v1.0)

### Step 2: Install the APK
1. Locate the APK file using **File Explorer**.
2. Tap to install.
3. If prompted, allow installation from **unknown sources**.
4. Bypass Play Protect warning if necessary.

---

## Features

### 1. Login System
- **Register a new account** (username, email, password).
- **Login with an existing account**.
- **Reset password** if forgotten.

### 2. Main Menu Functions
- **Pair Sensors** – Connect IoT devices.
- **Live Predict & Record** – Monitor real-time activity data.
- **Gather Raw Data** – Save sensor data for analysis.
- **View History** – Access past recorded data.

### 3. Sensor Pairing
- **Bluetooth Pairing**: Connect Thingy or RESpeck sensor.
- **QR Code Scanning**: Scan sensor ID for quick pairing.

### 4. Live Activity Recognition
- Uses **machine learning models** to predict **real-time gestures and breathing activities**.
- Displays recognized activities as **text and graphical icons**.
- Saves results as **CSV files** for future use.

### 5. Historical Data Viewing
- Browse saved activity records.
- **Integrated file viewer** for CSV data.
- Delete unnecessary records.

---

## Technical Details

### Classification Model Performance (On-device)
| Task | Accuracy |
|------|----------|
| Task 1 | **96.48%** |
| Task 2 | **85.71%** |
| Task 3 | **72.03%** |

### Supported Activities
- **Walking, Running, Sitting, Standing**
- **Lying Down (Various Positions)**
- **Stair Ascending/Descending**
- **Coughing, Hyperventilating**
- **Miscellaneous Movements**

### Software Architecture
- **Model-View-Controller (MVC) Structure**
- **Programming Language:** Kotlin
- **Database:** SQLite (future upgrade to Firebase)

### Machine Learning Model
- **CNN-based model** with a **4-layer architecture**.
- Uses **Batch Normalization, ReLU Activation, and Max Pooling**.
- Evaluated with **Leave-One-Subject-Out Cross-Validation (LOSOXV)**.

---

## Future Improvements
- Expand classification to **more human activities**.
- Integrate **cloud-based machine learning** for faster model training.
- Migrate to **Firebase** for real-time database syncing.
