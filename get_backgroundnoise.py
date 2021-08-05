#!/usr/bin/env python3
"""
Retrieve FSDKaggle2018 Test data.
"""

import os
import os.path


def get_background():
    if not os.path.exists("data"):
        os.makedirs("data/backgroundnoise")

    f = "https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_test.zip"
    os.system("cd data/backgroundnoise/ && wget -c %s" % repr(f))

    os.system(
        "cd data/backgroundnoise/ && unzip FSDKaggle2018.audio_test.zip && rm FSDKaggle2018.audio_test.zip"
    )


if __name__ == "__main__":
    get_background()
