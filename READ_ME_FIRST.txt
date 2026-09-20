================================================================
  DHANVANTARI VISION
  Chest radiograph screening triage
  Made by Team Akatsuki - Om, Aadidev, Subramanian, Parthiv
  Engineered by OMSR TECHNOLOGIES - omsrtech.com
================================================================

HOW TO RUN
----------
1. Unzip this folder anywhere (Desktop is fine).
2. Double-click  DhanvantariVision.exe
3. A black console window opens and the app opens in your web
   browser automatically. Give it 10-30 seconds the first time.
4. To stop, close the black console window.

Windows may show "Windows protected your PC" because the file is
not code-signed. Click "More info" then "Run anyway".

No installation, no Python, no internet connection required.


WHAT IT DOES
------------
It reads chest X-rays and sorts them so the films most likely to
show disease are read first. Two conditions are included, chosen
with the radio buttons at the top left:

  * Tuberculosis        trained on 800 real radiographs
  * Lung nodule / mass  trained on 2,214 real radiographs

Four tabs:
  Triage Worklist    the queue, sorted by risk
  Analyse a Scan     click a bundled film, or upload your own
  Model Report Card  the measured accuracy, honestly reported
  What This Is       the problem it addresses and its limits


A GOOD 2-MINUTE TOUR
--------------------
1. Open "Analyse a Scan".
2. Click a film under "Held-out TB" - the model flags it, and the
   heat map shows which part of the lung drove the decision.
3. Click one under "Held-out normal" - cleared, with a flat map.
4. Click one under "Pneumonia (different dataset)" - it still
   flags it, because the model has never been taught pneumonia.
   That is the honest limitation, shown deliberately.
5. Open "Model Report Card" to see where the model stops working.


IMPORTANT
---------
This is a student research prototype built on public research
datasets (US National Library of Medicine, and the NIH
ChestX-ray14 collection). It is NOT a medical device. It has no
clinical validation and no regulatory approval, and it must not
be used to make decisions about any real patient.

The AI-written note under each result is generated only from the
model's own numbers. If no internet connection is available it
falls back to a template and says so on screen.
