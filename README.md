# [Final Task] Prediction Model - Jordan Marcelino

## Domain Proyek

Proyek ini melibatkan perusahaan _multifinance_ sebagai klien. Tujuannya adalah untuk meningkatkan keakuratan penilaian dan pengelolaan risiko kredit, sehingga dapat mengoptimalkan keputusan bisnis dan meminimalkan potensi kerugian. Model machine learning akan dikembangkan untuk memprediksi risiko kredit menggunakan data pinjaman historis.

## Business Understanding

Untuk dapat mengoptimalkan keputusan bisnis dan meminimalkan potensi kerugian, proyek ini melakukan analisis terhadap dataset dengan pendekatan machine learning menggunakan algoritma klasifikasi.

### Problem Statements

-   Bagaimana efektivitas pendekatan _machine learning_ dalam memprediksi risiko kredit dilihat dari nilai metriks: akurasi, presisi, recall, dan f1-score?

### Goals

-   Mengevaluasi efektivitas pendekatan _machine learning_ dalam memprediksi risiko kredit dengan melihat nilai metriks: akurasi, presisi, recall, dan f1-score

    ### Solution statements

    -   Membandingkan 2 algoritma klasifikasi yaitu: Logistic Regression dan Random Forest. Kedua model akan diimprove dengan hyperparameter tuning menggunakan grid search dan cross validation dengan split sebanyak 5. f1-score digunakan sebagai nilai metrik utama dalam membandingkan masing-masing algoritma.

## Data Understanding

Data yang digunakan dalam proyek ini diambil dari [Rakamin LMS](https://rakamin-lms.s3.ap-southeast-1.amazonaws.com/vix-assets/idx-partners/loan_data_2007_2014.csv). Terdapat 1 dataset yang disediakan yaitu:

1.  loan_data_2007_2014.csv

### Variabel-variabel pada dataset adalah sebagai berikut:

| Nama kolom                  | Keterangan                                                                                                                                                                                                                                         |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| acc_now_delinq              | Jumlah rekening dimana peminjam sekarang menunggak.                                                                                                                                                                                                |
| addr_state                  | Negara yang disediakan oleh peminjam dalam permohonan pinjaman                                                                                                                                                                                     |
| all_util                    | Saldo hingga batas kredit pada semua perdagangan                                                                                                                                                                                                   |
| annual_inc                  | Pendapatan tahunan yang dilaporkan sendiri yang diberikan oleh peminjam saat pendaftaran.                                                                                                                                                          |
| annual_inc_joint            | Gabungan pendapatan tahunan yang dilaporkan sendiri yang diberikan oleh peminjam bersama pada saat pendaftaran                                                                                                                                     |
| application_type            | Menunjukkan apakah pinjaman tersebut merupakan permohonan perorangan atau permohonan bersama dengan dua peminjam bersama                                                                                                                           |
| collection_recovery_fee     | biaya pos dari biaya pengumpulan                                                                                                                                                                                                                   |
| collections_12_mths_ex_med  | Jumlah koleksi dalam 12 bulan tidak termasuk koleksi medis                                                                                                                                                                                         |
| delinq_2yrs                 | Jumlah tunggakan tunggakan lebih dari 30 hari dalam arsip kredit peminjam selama 2 tahun terakhir                                                                                                                                                  |
| desc                        | Deskripsi pinjaman yang diberikan oleh peminjam                                                                                                                                                                                                    |
| dti_joint                   | Rasio yang dihitung menggunakan total pembayaran bulanan peminjam bersama atas total kewajiban utang, tidak termasuk hipotek dan pinjaman LC yang diminta, dibagi dengan gabungan pendapatan bulanan yang dilaporkan sendiri oleh peminjam bersama |
| earliest_cr_line            | Bulan dimana batas kredit peminjam yang dilaporkan paling awal dibuka                                                                                                                                                                              |
| emp_length                  | Lamanya masa kerja dalam tahun. Nilai yang mungkin adalah antara 0 dan 10 dimana 0 berarti kurang dari satu tahun dan 10 berarti sepuluh tahun atau lebih.                                                                                         |
| emp_title                   | Jabatan yang diberikan oleh Peminjam saat mengajukan pinjaman.\*                                                                                                                                                                                   |
| funded_amnt                 | Jumlah total yang berkomitmen untuk pinjaman tersebut pada saat itu.                                                                                                                                                                               |
| grade                       | LC menetapkan tingkat pinjaman                                                                                                                                                                                                                     |
| home_ownership              | Status kepemilikan rumah yang diberikan oleh peminjam pada saat pendaftaran. Nilai-nilai kami adalah: SEWA, SENDIRI, KPR, LAINNYA.                                                                                                                 |
| id                          | ID unik yang ditetapkan LC untuk daftar pinjaman.                                                                                                                                                                                                  |
| il_util                     | Rasio total saldo saat ini terhadap kredit/batas kredit yang tinggi pada semua akun pemasangan                                                                                                                                                     |
| initial_list_status         | Status pencatatan awal pinjaman. Nilai yang mungkin adalah – Utuh, Pecahan                                                                                                                                                                         |
| inq_fi                      | Jumlah pertanyaan keuangan pribadi                                                                                                                                                                                                                 |
| inq_last_12m                | Jumlah pertanyaan kredit dalam 12 bulan terakhir                                                                                                                                                                                                   |
| inq_last_6mths              | Jumlah pertanyaan dalam 6 bulan terakhir (tidak termasuk pertanyaan otomotif dan hipotek)                                                                                                                                                          |
| installment                 | Pembayaran bulanan yang harus dibayar oleh peminjam jika pinjaman itu berasal.                                                                                                                                                                     |
| int_rate                    | Menunjukkan apakah pendapatan telah diverifikasi oleh LC, tidak diverifikasi, atau jika sumber pendapatan telah diverifikasi                                                                                                                       |
| issue_d                     | Bulan dimana pinjaman itu didanai                                                                                                                                                                                                                  |
| last_credit_pull_d          | Bulan terakhir LC menarik kredit untuk pinjaman ini                                                                                                                                                                                                |
| last_pymnt_amnt             | Jumlah total pembayaran terakhir yang diterima                                                                                                                                                                                                     |
| last_pymnt_d                | Pembayaran bulan lalu telah diterima                                                                                                                                                                                                               |
| loan_amnt                   | Pembayaran bulan lalu telah diterima                                                                                                                                                                                                               |
| loan_status                 | Status pinjaman saat ini                                                                                                                                                                                                                           |
| max_bal_bc                  | Saldo terhutang maksimum saat ini pada semua akun bergulir                                                                                                                                                                                         |
| member_id                   | LC unik yang menetapkan Id untuk anggota peminjam.                                                                                                                                                                                                 |
| mths_since_last_delinq      | Jumlah bulan sejak tunggakan terakhir peminjam.                                                                                                                                                                                                    |
| mths_since_last_major_derog | Bulan sejak rating 90 hari terakhir atau lebih buruk                                                                                                                                                                                               |
| mths_since_last_record      | Jumlah bulan sejak pencatatan publik terakhir.                                                                                                                                                                                                     |
| mths_since_rcnt_il          | Bulan sejak rekening cicilan terakhir dibuka                                                                                                                                                                                                       |
| next_pymnt_d                | Tanggal pembayaran terjadwal berikutnya                                                                                                                                                                                                            |
| open_acc                    | Jumlah jalur kredit terbuka dalam file kredit peminjam.                                                                                                                                                                                            |
| open_acc_6m                 | Jumlah perdagangan terbuka dalam 6 bulan terakhir                                                                                                                                                                                                  |
| open_il_12m                 | Jumlah perdagangan terbuka dalam 6 bulan terakhir                                                                                                                                                                                                  |
| open_il_24m                 | Jumlah rekening angsuran yang dibuka dalam 24 bulan terakhir                                                                                                                                                                                       |
| open_il_6m                  | Jumlah rekening angsuran yang dibuka dalam 12 bulan terakhir                                                                                                                                                                                       |
| open_rv_12m                 | Jumlah perdagangan bergulir yang dibuka dalam 12 bulan terakhir                                                                                                                                                                                    |
| open_rv_24m                 | Jumlah perdagangan bergulir yang dibuka dalam 24 bulan terakhir                                                                                                                                                                                    |
| out_prncp                   | Sisa pokok terutang untuk jumlah total yang didanai                                                                                                                                                                                                |
| out_prncp_inv               | Sisa pokok terutang untuk sebagian dari jumlah total yang didanai oleh investor                                                                                                                                                                    |
| policy_code                 | policy_code=1 yang tersedia untuk umum produk baru tidak tersedia untuk umum policy_code=2                                                                                                                                                         |
| pub_rec                     | Jumlah catatan publik yang menghina                                                                                                                                                                                                                |
| purpose                     | Kategori yang disediakan oleh peminjam untuk permintaan pinjaman.                                                                                                                                                                                  |
| pymnt_plan                  | Menunjukkan apakah rencana pembayaran telah diterapkan untuk pinjaman                                                                                                                                                                              |
| recoveries                  | Menunjukkan apakah rencana pembayaran telah diterapkan untuk pinjaman                                                                                                                                                                              |
| revol_bal                   | Total saldo kredit bergulir                                                                                                                                                                                                                        |
| revol_util                  | Tingkat pemanfaatan jalur bergulir, atau jumlah kredit yang digunakan peminjam relatif terhadap seluruh kredit bergulir yang tersedia.                                                                                                             |
| sub_grade                   | LC menetapkan tanah dasar pinjaman                                                                                                                                                                                                                 |
| term                        | Jumlah pembayaran pinjaman. Nilai dalam bulan dan dapat berupa 36 atau 60.                                                                                                                                                                         |
| title                       | Judul pinjaman yang diberikan oleh peminjam                                                                                                                                                                                                        |
| tot_coll_amt                | Total jumlah penagihan yang terutang                                                                                                                                                                                                               |
| tot_cur_bal                 | Total saldo saat ini dari semua akun                                                                                                                                                                                                               |
| total_acc                   | Jumlah total batas kredit yang saat ini ada dalam arsip kredit peminjam                                                                                                                                                                            |
| total_bal_il                | Total saldo saat ini dari semua rekening angsuran                                                                                                                                                                                                  |
| total_cu_tl                 | Jumlah perdagangan keuangan                                                                                                                                                                                                                        |
| total_pymnt                 | Pembayaran diterima sampai saat ini untuk jumlah total yang didanai                                                                                                                                                                                |
| total_pymnt_inv             | Pembayaran diterima hingga saat ini untuk sebagian dari jumlah total yang didanai oleh investor                                                                                                                                                    |
| total_rec_int               | Bunga yang diterima sampai saat ini                                                                                                                                                                                                                |
| total_rec_late_fee          | Biaya keterlambatan diterima sampai saat ini                                                                                                                                                                                                       |
| total_rec_prncp             | Kepala Sekolah diterima sampai saat ini                                                                                                                                                                                                            |
| url                         | URL untuk halaman LC dengan data daftar.                                                                                                                                                                                                           |
| zip_code                    | 3 angka pertama kode pos yang diberikan peminjam dalam permohonan pinjaman.                                                                                                                                                                        |

### EDA (Exploratory Data Analysis)

-   Deskriptif data

-   Menvisualisasikan distribusi label

-   Missing values

-   Univariate Analysis

-   Multiavariate Analysis

## Data Preparation

### Data preprocessing

1.  Penanganan missing values

2.  Membuang outlier

3.  Penanganan imbalanced label
    Karena terdapat ketidakseimbangan label pada data, maka akan digunakan teknik _over sampling_ dengan metode SMOTE. Kegunaannya adalah agar model dapat mempelajari dengan baik perbedaan antar label.

4.  Splitting data
    Data akan displit menjadi data latih dan data uji dengan data uji sebesar 10%. Data latih akan digunakan untuk melatih model, sedangkan data uji akan digunakan untuk mengukur performa model pada data yang belum pernah dilihat sebelumnya.

5.  Scaling data
    Selanjutnya data akan distandarisasi menggunakan Standard Scaler, dimana scaler akan dilatih pada data latih saja, lalu diterapkan ke data uji, sehingga menghindari adanya data leakage.

### Modeling

-   Tahapan pertama adalah data preprocessing, di mana data dari dataset dipersiapkan untuk digunakan dalam pembuatan model. Untuk lebih detail sudah dijelaskan pada bagian [data preparation](#data-preparation)

-   2 Algoritma klasifikasi, yaitu; Logistic Regression dan Random Forests, akan dilakukan hyperparameter tuning kemudian dibandingkan menggunakan cross-validation dengan split = 5. Metrik evaluasi yang digunakan adalah f1 score, model dengan rata-rata nilai f1 score terbesar akan menjadi model terbaik.

### Evaluation

Model akan dievaluasi pada test set untuk memastikan bahwa model tidak overfit, dan dapat memprediksi data yang belum pernah dilihat secara akurat. Terdapat 4 metrik evaluasi yang digunakan:

-   $ Accuracy = \frac{(TP+TN)}{(TP+TN+FN+FP)} $

-   $ Precision = \frac{TP}{(TP+FP)} $

-   $ Recall = \frac{TP}{(TP+FN)} $

-   $ F1 = \frac{(2*precision*recall)}{(precision+recall)} $

Keterangan:

-   TP (True Positive) = Prediksi 1, Ground Truth 1
-   TN (True Negative) = Prediksi 0, Ground Truth 0
-   FP (False Positive) = Prediksi 1, Ground Truth 0
-   FN (False Negative) = Prediksi 0, Ground Truth 1
