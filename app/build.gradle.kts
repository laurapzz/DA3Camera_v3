plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.da3camera"
    compileSdk {
        version = release(37)
    }
    ndkVersion = "26.1.10909125"

    buildFeatures {
        prefab = true
    }

    androidResources {
        noCompress += "pte"
        noCompress += "onnx"
    }

    defaultConfig {
        applicationId = "com.example.cameradepthlane.v3"
        minSdk = 28
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters.addAll(listOf("arm64-v8a", "x86_64"))
        }

        externalNativeBuild {
            cmake {
                cppFlags("-std=c++17")
                arguments("-DANDROID_STL=c++_shared")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            optimization {
                enable = true
                packageScope = setOf("androidx.**", "kotlin.**", "kotlinx.**")
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    packaging {
        jniLibs {
            pickFirsts += "**/libc++_shared.so"
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.1")
    implementation("org.opencv:opencv:4.10.0")
    implementation("org.pytorch:executorch-android:1.5.1")
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    // CameraX
    val cameraxVersion = "1.4.2"

    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    implementation("androidx.exifinterface:exifinterface:1.4.1")
}
