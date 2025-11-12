# Dokumentasi Perintah SQLGenX

Berikut adalah panduan lengkap untuk semua perintah yang tersedia di `sqlgenx`. Setiap perintah dirancang untuk membantu Anda mengelola _workspace_ dan menghasilkan kueri SQL dengan mudah.

---

## `sqlgenx load`

Memuat sebuah file skema SQL ke dalam sebuah _workspace_ baru. Perintah ini akan membuat direktori _workspace_, mem-parsing file skema, membuat _vector embeddings_ untuk pencarian kontekstual, dan secara otomatis mengaktifkan _workspace_ yang baru dibuat.

### Penggunaan

```bash
sqlgenx load [OPTIONS] SCHEMA_FILE
```

### Argumen

| Argumen       | Deskripsi                              |
| ------------- | -------------------------------------- |
| `SCHEMA_FILE` | **(Wajib)** Path ke file skema `.sql`. |

### Opsi

| Opsi     | Alias | Deskripsi                                                                                                  |
| -------- | ----- | ---------------------------------------------------------------------------------------------------------- |
| `--name` | `-n`  | Nama yang akan diberikan untuk _workspace_. Jika tidak disediakan, nama akan diambil dari nama file skema. |
| `--dbms` | `-d`  | Tipe database (misalnya, `mysql`, `postgresql`, `sqlite`). Defaultnya adalah `generic`.                    |
| `--desc` |       | Deskripsi singkat untuk _workspace_.                                                                       |

### Contoh

```bash
# Memuat skema e-commerce dengan nama 'ecommerce_db' dan tipe DBMS 'mysql'
sqlgenx load examples/ecommerce_schema.sql --name ecommerce_db --dbms mysql --desc "Database untuk toko online"

# Memuat skema dengan nama yang dihasilkan secara otomatis
sqlgenx load /path/to/my_schema.sql --dbms postgresql
```

---

## `sqlgenx generate`

Menghasilkan kueri SQL dari perintah bahasa alami. Perintah ini menggunakan _workspace_ yang aktif untuk mengambil konteks skema yang relevan, lalu mengirimkannya ke LLM bersama dengan permintaan Anda untuk membuat kueri SQL yang akurat.

### Penggunaan

```bash
sqlgenx generate [OPTIONS] "QUERY"
```

### Argumen

| Argumen   | Deskripsi                                  |
| --------- | ------------------------------------------ |
| `"QUERY"` | **(Wajib)** Permintaan dalam bahasa alami. |

### Opsi

| Opsi          | Alias | Deskripsi                                                                   |
| ------------- | ----- | --------------------------------------------------------------------------- |
| `--workspace` | `-w`  | Menentukan _workspace_ mana yang akan digunakan untuk generasi kueri.       |
| `--explain`   | `-e`  | Memberikan penjelasan dalam bahasa alami tentang kueri SQL yang dihasilkan. |
| `--copy`      | `-c`  | Menyalin kueri SQL yang dihasilkan ke _clipboard_.                          |

### Contoh

```bash
# Menghasilkan kueri sederhana
sqlgenx generate "Tunjukkan 5 pelanggan teratas berdasarkan jumlah total pesanan"

# Menghasilkan kueri dengan penjelasan
sqlgenx generate "Produk apa yang paling banyak terjual bulan ini?" --explain

# Menggunakan workspace spesifik dan menyalin hasilnya
sqlgenx generate "Daftar semua kategori produk" --workspace ecommerce_db --copy
```

---

## `sqlgenx list`

Menampilkan daftar semua _workspace_ yang tersedia dalam bentuk tabel yang rapi, beserta informasi seperti DBMS, tanggal terakhir digunakan, dan _workspace_ mana yang sedang aktif.

### Penggunaan

```bash
sqlgenx list
```

### Contoh

```bash
sqlgenx list
```

---

## `sqlgenx use`

Beralih ke _workspace_ lain, menjadikannya _workspace_ aktif untuk perintah `generate` selanjutnya.

### Penggunaan

```bash
sqlgenx use WORKSPACE
```

### Argumen

| Argumen     | Deskripsi                            |
| ----------- | ------------------------------------ |
| `WORKSPACE` | **(Wajib)** Nama _workspace_ tujuan. |

### Contoh

```bash
sqlgenx use ecommerce_db
```

---

## `sqlgenx info`

Menampilkan informasi detail tentang _workspace_ tertentu (atau yang sedang aktif jika tidak ditentukan), termasuk metadata dan daftar tabel yang terdeteksi.

### Penggunaan

```bash
sqlgenx info [OPTIONS]
```

### Opsi

| Opsi          | Alias | Deskripsi                                         |
| ------------- | ----- | ------------------------------------------------- |
| `--workspace` | `-w`  | Nama _workspace_ yang ingin dilihat informasinya. |

### Contoh

```bash
# Menampilkan info untuk workspace yang sedang aktif
sqlgenx info

# Menampilkan info untuk workspace spesifik
sqlgenx info --workspace ecommerce_db
```

---

## `sqlgenx delete`

Menghapus _workspace_ beserta semua datanya, termasuk file skema yang disalin, _vector embeddings_, dan metadata.

### Penggunaan

```bash
sqlgenx delete [OPTIONS] WORKSPACE
```

### Argumen

| Argumen     | Deskripsi                                       |
| ----------- | ----------------------------------------------- |
| `WORKSPACE` | **(Wajib)** Nama _workspace_ yang akan dihapus. |

### Opsi

| Opsi      | Alias | Deskripsi                                     |
| --------- | ----- | --------------------------------------------- |
| `--force` | `-f`  | Melewatkan permintaan konfirmasi penghapusan. |

### Contoh

```bash
# Menghapus workspace dengan konfirmasi
sqlgenx delete old_workspace

# Menghapus workspace secara paksa tanpa konfirmasi
sqlgenx delete temporary_ws --force
```
