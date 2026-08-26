"""
Controller layer: menyusun L1-L4 jadi satu alur yang bisa dijalanin per nasabah.

Isinya nggak ada logika domain sama sekali. Semua keputusan soal fraud tetep tinggal di
`src/detection`, `src/investigation`, `src/decision`, dan `src/reporting` -- modul di sini
cuma nentuin siapa jalan setelah siapa dan data apa yang dioper.
"""
