# KnotVision Pipeline v6.4

Müşteri fotoğrafı → Texcelle'de açılan, makinenin renk sınırına uyan, üretime giden indeksli desen dosyası.

Kapsam: Wilton, parça halı. Desen üretimi iki mod: `segment` (deterministik ağ, ızgara çözünürlüğünde, **varsayılan**) ve `quantize` (modelsiz, taban çizgisi, `distressed` sınıfının varsayılanı, `segment` hata verirse yedek).

Kapsam dışı:
- Retrieval, rulo tekrar, `pictorial` sınıfı (manzara, portre, foto-halı), Wilton dışı tezgahlar — v7.
- Diffusion tabanlı `edit` modu — v7 adayı (bkz. Sadeleştirme kararları).
- **İşlem iplikleri** (oyma, kesme, parlaklık). İşlem bir iplik renginin tamamına uygulanır; Texcelle'de desenci tarafından ipliğe etiket olarak girilir. Desen dosyasında ayrı indeks gerektirmez, fotoğraftan tespit edilmez. Pipeline'ın işi deseni doğru renkle çizmektir.

Pipeline'ın davranışı **desen sınıfı** (`design_class`) ile belirlenir; sınıfın kaynağı gerçek işlerde desencidir (bkz. Desen sınıfı). Sınıf; temizleme eşiklerini, düzenlileştirmenin açık/kapalı olmasını, zorlama bayraklarını ve QC özellik setini seçer. Dither ise sınıftan değil, gradyan maskesinin varlığından türer.

Hedef: fotoğrafa olabildiğince yakın, **desenci elinden çıkmış gibi düzenli** desen. Kalite tanımı dört parçadır ve dördü de ölçülür:
1. Sadakat — motifler doğru yerde, doğru renkte, yoktan motif eklenmemiş, var olan motif silinmemiş.
2. Düzenlilik — çizgiler düz, diagonaller temiz basamaklı, tekrar eden motifler birebir aynı, simetri birebir. `regularize` üretir; `line_jaggedness`, `repeat_mismatch` ölçer.
3. Renk geçişi — gradyanlı bölgeler bantlaşmadan, desencinin dither düzeniyle. `dither` üretir; `banding`, `dither_err` ölçer.
4. Gürültü sadakati — eskitme desenlerinde kasıtlı gürültü korunur, fotoğraf kaynaklı gürültü (hav dokusu, iplik sapması, kamera/JPEG) düğümlere yazılmaz. `noise_fidelity` ölçer (eval), `noise_1knot_pct` izler (QC).

Pipeline çıktısı desencinin düzelttiği bir taslak değil, üretime giden nihai dosyadır. Yoktan motif eklenmesi ve var olan motifin silinmesi kabul edilmez. Yoktan motif için eval'de `cc_added_motif = 0` sert kural; motif silinmesi için eval'de `cc_missing_pct` ve her işte çevrimiçi `guard`. İki ayrı kontrol vardır:
1. Offline değerlendirme — model iyi mi (seyrek, referanslı).
2. Online kalite kontrolü — bu iş desenciye gösterilmeye değer mi (her iş, referanssız).

Lisans kuralı: her model ve kütüphane ticari kullanımda ücretsiz olmalı. Kabul: Apache 2.0, MIT, BSD, SAM License. Ret: AGPL (ultralytics), "research only" ağırlıklar.

---

## Alan bilgisi ve tasarım kararları

| Konu | Bilgi | Pipeline'daki karşılığı |
|---|---|---|
| İşlem | Texcelle'de derinlik katmanı yoktur; işlem ipliğe bağlı etikettir, rengin tamamına uygulanır. | Kapsam dışı. `palette_used ≤ machine_colors`. |
| Izgara | Wilton'da yatay yoğunluk (tarak = çözgü/m) tezgah sabitidir; dikey (atkı/m) iş bazında ayarlanır. Yoğunluğun tanımı fabrikaya göre değişir. | `reed_per_m`, `density_definition` tezgah profilinde; üç tanımın formülü `intake`'te. Fotoğrafta düğüm dokusu görünüyorsa `knot_grid_field` tanımdan bağımsız çapraz kontrol eder. |
| Saçak | Sonradan dikilir, desen dışıdır; fotoğrafta vardır. | `rectify`'da saçak tespiti; maske hav kenarına kırpılır. |
| Arşiv desenleri | Gradyanlı desenler 8–12 düz renkle kasıtlı dither içerir. Eskitme desenlerinde gürültü kasıtlıdır. Aynı RGB'li birden fazla indeks bulunabilir. | Dither düzeni arşivden (`tools/dither_stats.py`), gürültü ölçeği arşivden (`tools/noise_stats.py`); ikisi de desenciye teyit ettirilir. Sentetik üreteç aynı RGB'li indeksleri birleştirir. |
| Gradyan | Modern desenlerin çoğunda renk geçişi vardır; desenci düz renklerle dither çizer. En yakın indeks ataması bantlaşma üretir; ΔE metriği bunu yakalamaz. | Gradyan maskesi (`structure`), maske içinde analitik dither, `banding` QC özelliği. Maske içinde temizleme, regularize ve komşuluk cezası kapalı. Maske sınıftan bağımsız çalışır. |
| Gürültü kaynağı | Fotoğraftaki gürültü üç şeyin toplamıdır: kasıtlı eskitme (istenen), hav dokusu + iplik sapması (istenmeyen), kamera/JPEG (istenmeyen). Kasıtlı eskitme düğüm ızgarasına oturur ve çok düğümlük lekeler halindedir; fotoğraf gürültüsü tek düğüm ölçeğinde rastgele dağılır. | `noise_frac` tek düğümlük bileşenleri saymaz. `noise_1knot_pct` ayrı ölçülür. `distressed`'ta tek düğümlük adalar temizlenir (`min_region_knots: 1`), daha büyükleri korunur; arşivde tek düğümlük kasıtlı gürültü bulunursa `noise_stats.py` bunu 0'a çeker. Bulanık fotoğrafta (`pile_blur`) yüksek `noise_frac` fotoğraf gürültüsü sayılır, `distressed` kararı verilmez. |
| Parlama / hayalet motif | Renk farkıyla değil hav yatışı veya işlemle görünen motif fotoğraftan ayırt edilemez: ayrı iplik ama RGB'si yakın (palet birleştirir, motif kaybolur), tek renge işlem (palete boşa indeks açılır) veya sadece hav artefaktı (yoktan motif çizilir). Aydınlatma düzleştirmesi bunu çözmez; parlama motif şeklini izler, düşük frekanslı değildir. | Otomatik çözüm yoktur. `score_sheet.csv`'de `fix_type: sheen`; desenci ölçümünde sıklığı ölçülür. Sıklık yüksekse v7'de ele alınır. |
| Fotoğraf çözünürlüğü | Eskitme ve detaylı desenler 1–2 düğüm ölçeğinde bilgi taşır. 160×230 cm, 400 çözgü/m → 640×920 düğüm → 4 px/düğüm için 2560 px genişlik; bulanık kenar efektif değeri düşürür. | Müşteri akışına ve çift toplamaya **uzun kenar ≥ 3000 px** şartı (`photo_long_px_min`), önerilen 4000. Asıl kapı efektif çözünürlüktür (`photo_px_per_knot_eff`). |
| Sapma | Desencinin nihai deseni fotoğraftan sapabilir; saha bilgisi "çok küçük". | `align_report.csv` ile ölçülür; `cc_added_motif = 0` sert kural. |
| Desenci pratiği | Düz çizgi çizer, aynı motifi birebir kopyalar, diagonalleri temiz basamaklandırır. Bordür köşesi kenar şeridinin periyoduna uymayan ayrı motiftir. | `regularize`; köşe blokları muaf; `line_jaggedness`, `repeat_mismatch`. |
| İplik RGB | Desenci paleti kendi seçer, Texcelle'de gerçek ipliklerle eşler. Kütüphane RGB'leri yaklaşıktır. | İplik kütüphanesi opsiyonel. Palet her durumda öneridir (`palette_confirmed`). |
| Hav yönü / ışık | En büyük renk hatası havın yatış yönüdür. | `rectify`'da düşük frekanslı çarpımsal aydınlatma düzleştirme; sentetikte hav yönü bağımlı gölgelenme. |
| Uzun hav / odak | Motif kenarı 3–4 düğüme yayılır; piksel/düğüm yüksek olsa bile efektif çözünürlük düşüktür. | `edge_width_knots`, `photo_px_per_knot_eff`, `pile_blur`. |
| Regularize kazancı | Klasik motif (yaprak, rumi, palmet) çoğunlukla serbest parçaya düşer; açı kilidi ve çember fit asıl kazancı geometrikte, bordür çerçevesinde ve diagonalde verir. | `free_len_share` sınıf kırılımıyla raporlanır; klasikte > 0.6 ise v7 öğrenilmiş vektörleştirme gündeme alınır. |

---

## Kurulum

Fabrikadan fabrikaya ve desenciden desenciye değişen her şey config'dir. Yeni kurulumda üç dosya ve bir arşiv taraması:

| Ne | Dosya | Kaynak | Yoksa (pipeline yine çalışır) |
|---|---|---|---|
| Tezgah | `config/looms/<fabrika>.yaml` — `reed_per_m`, `density_definition`, `allowed_machine_colors` | fabrika | `reed_per_m: null` → kare varsayımı, `grid_unverified` sert kapı; ilk işteki `knot_grid` ölçümü profili doldurur |
| Dither düzeni | `config/dither/<ad>.yaml` | `tools/dither_stats.py` arşivden, desenci teyit eder | 4×4 ordered, `dither_pattern_default` sert kapı (sadece gradyan maskesi olan işlerde) |
| İplik kütüphanesi | `config/yarns/<desenci>.csv` | desenci | palet fotoğraftan, `yarn_code` boş |
| Motif eşiği | profilde `motif_min_knots` | `tools/motif_stats.py` arşivden | 10, `motif_min_knots_source: assumed` |
| Gürültü ölçeği | profilde `noise_cc_min`, `noise_cc_max`, distressed `min_region_knots` | `tools/noise_stats.py` arşivdeki eskitme desenlerinden | 2 / 6 / 1, `noise_scale_source: assumed` |
| QC kalibrasyonu | `models/qc_calib.json` | `eval/calibrate_qc.py` | skor hesaplanır, karar verilmez (`needs_review`) |

Kurulum eksikliği sistematik `needs_review` üretir: `reed_per_m: null` her işte, `dither_pattern: null` gradyan maskesi olan her işte, `qc_calib.json` yokluğu her işte. Bunlar dolmadan otomatik kabul yoktur; `cli.py setup-check` eksik alanları ve her birinin hangi kapıyı tetiklediğini listeler.

Kurulum durumu her işin raporunda `setup` bloğuna yazılır: `{reed_per_m: profile | assumed, density_definition, dither_pattern: designer | archive | default, yarn_library: bool, motif_min_knots: archive | assumed, noise_scale: archive | assumed, qc_calib: bool}`.

SAM 3.1 erişimi (HF gated) kurulumdan bağımsızdır; yoksa `corners_override` ile manuel köşe.

---

## Desen sınıfı (`design_class`)

`design_class ∈ {classic, geometric, gradient, distressed, minimal}`. `pictorial` tespit edilirse iş `class_unsupported` ile durur (desenci `--class <c>` ile ezerse `needs_review` ile devam eder).

Kaynak:
- **Gerçek iş:** desenci girdisi zorunlu (`class_source: designer`). Otomatik tespit yine çalışır; sadece `class_auto`, `class_conf` ve `class_features` raporlanır. Otomatik sınıf desenciyle çelişiyor ve `class_conf ≥ class_conf_min` ise `warnings += "class_conflict"` (yumuşak; karar desencinindir).
- **Sentetik veri ve toplu değerlendirme:** `--class auto` (`class_source: auto`). `class_conf < class_conf_min` → `class_uncertain`, sert kapı `needs_review`.

Sınıflar gerçek halılarda birbirini dışlamaz (degrade zeminli madalyon, gradyanlı kafes, hem eskitmeli hem gradyanlı modern). Çoklu etiket **kullanılmaz**: parametre katmanı tek sınıf dosyası okur; iki dosyanın çakışma tablosu v6.2'de kaldırılan karmaşıklığı geri getirir. Bunun yerine sınıftan bağımsız çalışan mekanizmalar karışık halıyı taşır: gradyan maskesi (dither, maske içi muafiyetler, palette uç renkleri) sınıfa bakmaz. Sınıfa bağlı kalan tek şey `design_mode` ve maske dışındaki `min_region_knots` / `min_coverage_pct` / `regularize` / `force_*`. `gradient_frac` ve `noise_frac` ikisi birden eşik üstündeyse `warnings += "class_mixed"`, sert kapı `needs_review` (hangi kaynaktan gelirse gelsin).

### Parametre katmanı

Her parametre üç katmandan çözülür, sonraki öncekini ezer: tezgah profili ← sınıf dosyası (`config/classes/<design_class>.yaml`) ← iş override'ı. Çözülmüş set `job.json.resolved_params`; hash'i aşama cache anahtarına girer. Sınıf dosyaları `allowed_machine_colors`, `reed_per_m`, `density_definition`, `retention_days` dışında her alanı ezebilir.

### Sınıfların değiştirdiği parametreler

| Sınıf | Sınıf katmanı olmasa kırıldığı yer | Sınıf dosyasının ezdiği alanlar |
|---|---|---|
| `classic` | — | varsayılanlar |
| `geometric` | — (regularize burada en iyi çalışır) | varsayılanlar; `force_symmetry: auto` kalır |
| `gradient` | maske dışı davranış varsayılanla aynı; sınıf sadece QC özelliğini ekler ve raporlar | `qc_features_extra: [banding]` (maske içi muafiyetler maskeden türer, sınıf dosyasında değildir) |
| `distressed` | temizleme, kapsama birleştirme ve regularize kasıtlı gürültüyü siler; `noise_knots_pct` yüksek çıkar ve doğru çıktı reddedilir; segment boundary kanalı doyar | `design_mode: quantize`, `regularize: off`, `min_region_knots: 1` (sadece tek düğümlük adalar), `min_coverage_pct: 0`, `force_*: off`; `noise_knots_pct`, `cleanup_changed_pct` null |
| `minimal` | kenar yokluğunda `rerender_edge_iou` ve `line_jaggedness` gürültü | `edge_density < edge_density_min` ise bu iki özellik null; `force_symmetry: off` |

Panel / çerçeveli desen ayrı sınıf değildir; bordür kabul koşuluyla ele alınır (`structure` adım 2).

### Otomatik tespit (`structure` adım 6)

Girdi büyüklükleri `structure`'ın zaten hesapladıklarıdır:
- `edge_density`: `rectified_contrast` (512 px) Canny kenar piksel oranı.
- `n_colors_est`: 400 px Lab k-means (k = 16, sabit seed), merkezler ΔE < 8 ise birleştirilir, kalan küme sayısı.
- `gradient_frac`: gradyan maskesinin alan oranı (adım 5b).
- `noise_frac`: ızgara çözünürlüğündeki kaba kuantizasyonda (adım 9'daki etiket haritası) alanı ∈ `[noise_cc_min, noise_cc_max]` düğüm olan bileşenlerin düğüm oranı. Tek düğümlük bileşenler **sayılmaz**; onlar `noise_1knot_frac`'a gider. Gerekçe: kasıtlı eskitme çok düğümlük lekedir, tek düğüm fotoğraf gürültüsüdür.
- `symmetry_conf`, `border_period_conf`, `ground_repeat.conf`, `medallion`.
- `line_frac`: kenar piksellerinin `HoughLinesP` doğrularıyla kapsanan oranı.

Kurallar sırayla, ilk eşleşen; güven = eşleşen kuralın skoru ile bir sonraki en yüksek kuralın skoru arasındaki fark:
1. `gradient_frac ≥ 0.6` ve tekrar/simetri yok → `pictorial`.
2. `noise_frac ≥ noise_class_frac` (0.08) ve `pile_blur` yok → `distressed`.
3. `gradient_frac ≥ gradient_class_frac` (0.25) → `gradient`.
4. `n_colors_est ≤ 3` ve `edge_density < edge_density_min` → `minimal`.
5. `symmetry == hv` veya (bordür kabul + `medallion`) → `classic`.
6. `ground_repeat` tespit veya `line_frac ≥ 0.6` → `geometric`.
7. aksi → `classic`.

Sıra gerekçesi: hem gradyanlı hem gürültülü halıda yanlış `gradient` kararı gürültüyü siler (geri alınamaz); yanlış `distressed` kararında gradyan maskesi ve dither yine çalışır, sadece mod `quantize` olur. Ucuz hata yönü öne alınmıştır. `pile_blur` koşulu: bulanık fotoğrafta yüksek `noise_frac` desen gürültüsü değildir.

Rapor: `design_class`, `class_source`, `class_auto`, `class_conf`, `class_features`, `class_conflict`, `class_mixed`.

---

## Motif koruma (`guard.py`)

İndeks haritasını değiştiren her aşamanın sonunda: `assert_no_motif_loss(before, after, params) → after'`.
- `before` ve `after` bileşenleri eşleştirilir (aynı indeks, IoU ≥ 0.3).
- `before`'da alanı > `motif_min_knots` olan ve `after`'da eşleşmeyen her bileşen geri yazılır. Aşama geri alınmaz, sadece bileşen.
- Gradyan maskesi ve `distressed`'ta harita kontrol dışıdır.
- Çağrı noktaları: `production` temizleme sonrası, `production` kapsama birleştirmesi sonrası, `regularize` adım 2 (her yapı sonrası), `regularize` adım 3. `palette`'te indeks haritası yoktur; koruma `motif_color_min_cc` ile kalır.
- Her çağrı `stages/guard.json`'a eklenir (`stage, reverted_components, reverted_area_pct`). Dosya `production` ve `regularize`'ın çıktısıdır, `qc` girdisidir.
- Toplam `reverted_area_pct > guard_warn_pct` (0.02) → `guard_heavy`, sert kapı.
- Eval'de `cc_missing_pct` aşama kırılımıyla raporlanır; hangi aşamanın motif sildiği buradan okunur.

Regularize için ayrı "değişen düğüm yüzdesi" kapısı yoktur: yüzde kapısı düzenlileştirmeye en çok ihtiyaç duyan işi engeller; guard asıl riski (motif silinmesi) doğrudan ölçer.

`motif_min_knots`: `tools/motif_stats.py` arşiv truth desenlerinde `is_thin` olmayan bileşen alanı dağılımının 5. yüzdeliği. Arşiv yoksa 10, `assumed`.

---

## Kabul ve kontrol

### 1. Offline değerlendirme (model geliştirme)

Soru: model iyi mi. Seyrek çalışır: yeni checkpoint, veri değişimi. Referans `truth.tiff`.

Desenci puanı ve **düzeltme süresi** kalibrasyon verisidir. Puan: 5 = değişiklik yapmadan üretime; 4 = < 30 dk; 3 = < 2 saat; 2 = sadece şablon; 1 = sıfırdan. `score_sheet.csv`: puan, düzeltme süresi (dk), düzeltme türü (`geometry | color | regularity | motif | gradient | noise | sheen`), `design_class`.

Yayın kriteri, **iki sabit sette ayrı ayrı** (`test_core_classic`, `test_core_modern`) + 100 sabit sentetik (arşiv) + 100 sabit sentetik (prosedürel), her çiftte:
- `cc_added_motif = 0` (sert; gradyan maskesi ve truth gürültü maskesi dışında).
- `cc_added_area_pct ≤ 0.002`.
- `cc_missing_pct ≤ 0.01`.
- `thin_line_recall ≥ 0.9`.
- `knot_deltaE_mean < deltaE_warn`, `knot_deltaE_p95 < 2 × deltaE_warn` (gradyan maskesinde 8×8 blok ortalaması).
- `line_jaggedness_delta ≤ jag_max` (regularize açık işlerde; başlangıç 0.15, quantize taban çizgisi sonrası sabitlenir).
- `repeat_mismatch ≤ 0.01` tekrar bölgelerinde.
- `banding ≤ banding_max` gradyan maskesi olan işlerde (başlangıç 2.0 L).
- `noise_fidelity ≥ noise_fidelity_min` truth gürültü maskesi olan işlerde (başlangıç 0.8).
- Online QC skorunun her iki `test_core` ortalaması ≥ `qc_accept`.
- Önceki sürüme göre hiçbir metrikte, hiçbir sınıfta > %10 kötüleşme (CI, sınıf kırılımlı).

`deltaE_warn` başlangıç 3.0 ve yer tutucudur; 5.0 (önceki değer) düğüm başına gözle görülür kaymadır. Desenci ölçümünde `fix_type: color` ile korelasyona göre sabitlenir.

### 2. Online kalite kontrolü (her iş, `qc` aşaması)

Soru: bu çıktı desenciye gösterilmeye değer mi. Referans yok.

**Sert kapılar** (kalibrasyon gerektirmez):
- `palette_used ≤ machine_colors`.
- `grid_unverified`, `grid_mismatch`, `partial_view`, `non_planar`, `non_planar_internal` (düzeltilemediyse), `internal_geometry: unmeasurable`, `forced_low_res`, `pile_blur`, `dither_pattern_default`, `class_uncertain`, `class_unsupported`, `class_mixed`, `guard_heavy` → `needs_review`.
- `(design_class, design_mode, regularize)` üçlüsü `qc_calib.json.calibrated_configs` içinde yoksa → `mode_uncalibrated`, `needs_review`. Kalibrasyonda görülmemiş konfigürasyon için karar verilmez; sınıf varsayılanı olup olmaması ölçüt değildir.
- `qc_calib.json` yoksa veya `qc_calib_stale` → `needs_review` (skor hesaplanır, karar verilmez).

**Yumuşak özellikler.** Ortak set QC modelinin girdisidir; null değer modele girmez; kalibrasyon setinde > %50 null olan özellik modelden çıkarılır. `design_class` kategorik değişkendir.

| Özellik | Tanım | Null kuralı | Sinyal |
|---|---|---|---|
| `rerender_edge_iou` | `edge_fidelity(rerender, rectified)`: guided filter (r=8) + Lab L Canny, 2 px tolerans kenar IoU. | `edge_density < edge_density_min` | Motif konumu |
| `rerender_color_deltaE` | Render ve `rectified`, 16×16 blok Lab ortalamaları arası ΔE2000 ortalama ve p95. | — | Renk |
| `palette_deltaE_p50`, `_p95` | `rectified` piksellerinin en yakın palet girişine ΔE2000 (gradyan maskesinde dither karışım rengine). | — | Palet yetersizliği |
| `symmetry_consistency` | `rectified` ve desenin simetri skorları farkı. | simetri yok | Simetri |
| `repeat_mismatch_raw` | `regularize` öncesi, tekrar bölgelerinde periyotlar arası uyuşmazlık. | tekrar yok | Tekrar tutarlılığı |
| `line_jaggedness` | Sınır zincirlerinde 16 düğümlük pencerede run-length entropisi, gradyan maskesi dışı. | `edge_density < edge_density_min` | Çizgi düzensizliği |
| `noise_knots_pct` | Temizleme öncesi alanı ≤ `min_region_knots` bileşenlerin düğüm oranı. | `distressed` | Gürültü |
| `noise_1knot_pct` | Temizleme öncesi tek düğümlük bileşenlerin düğüm oranı, gradyan maskesi dışı. | — | Fotoğraf gürültüsü (her sınıfta hesaplanır; `distressed`'ta yüksekse çıktı fotoğraf dokusunu taşıyor demektir) |
| `cleanup_changed_pct` | Temizleme ile değişen düğüm oranı. | `distressed` | |
| `regularize_changed_pct` | `regularize` ile değişen düğüm oranı. | `regularize: off` | Ham düzensizlik |
| `guard_reverted_pct` | Guard'ın geri yazdığı düğüm oranı. | — | Aşama/yapı çelişkisi |
| `banding` | Gradyan maskesinde geri-render ile `rectified` arasındaki L farkının düşük frekanslı (σ = 8 düğüm) RMS'i. | maske yok | Bantlaşma |
| `edge_width_knots` | Kenar geçiş genişliği, düğüm. | — | Hav bulanıklığı |
| `wb_confidence` | Gray-world kazanç sapması, kırpılmış parlak piksel oranı. | — | Renk kalibrasyonu |
| `illum_range`, `illum_residual` | Aydınlatma alanı genliği; düzleştirme sonrası kalan gradyan. | — | Işık |

Rapor amaçlı (modele girmez): `model_uncertainty` (segment: `0.3 < boundary < 0.7` oranı, maske dışı).

**Skor:** proportional-odds ordinal lojistik regresyon (`statsmodels.OrderedModel`), sayısal özellikler standardize, `design_class` one-hot (≥ 10 örneği olan sınıflar ayrı düzey, kalanlar `other`), L2. `qc_score = P(puan ≥ qc_target)`; `qc_target ∈ {5, 4, 3}`, hem ≥ t hem < t sınıfında ≥ 15 örnek olan en yüksek t. `P(puan = 5)` ayrıca (`qc.p5`).

**Karar:** `qc_score ≥ qc_accept` → `accepted`; `< qc_reject` → `rejected`; arada → `needs_review`. `qc.status` hedefle yazılır: `accepted@4`. `qc.reasons`: skoru en çok düşüren 3 özellik.

**Kalibrasyon protokolü (`eval/calibrate_qc.py`):**
1. ≥ 100 çıktı, sınıf kotası: ≥ 15 `classic`, 15 `geometric`, 10 `gradient`, 10 `distressed`, 5 `minimal`; kota dolmayan sınıf `other`'a düşer ve o sınıfta karar verilmez. Her çıktı sınıfının varsayılan konfigürasyonuyla; kalibre edilen konfigürasyonlar `calibrated_configs`'a yazılır. Desenci kör puanlar + süre + tür.
2. Değerlendirici tutarlılığı: 10 çıktı ≥ 2 hafta sonra ikinci kez; ağırlıklı kappa. κ < 0.5 → tüm set iki kez puanlanır, ortalama etiket olur.
3. Ortak özellik seti hesaplanır.
4. Ordinal model, 5-kat çapraz doğrulama; AUC toplam ve sınıf başına. Toplam < 0.75 → red. Sınıf < 0.65 → `class_unreliable`, o sınıfta `needs_review`.
5. `qc_accept`: puan ≥ hedef olanları ≥ 0.9 kesinlikle yakalayan en düşük eşik. `qc_reject`: puan ≤ 2 olanların ≥ %90'ını düşüren en yüksek eşik. Profildeki 0.8 / 0.3 kalibrasyona kadar rapor amaçlıdır.
6. Yeniden kalibrasyon: model hash değişince zorunlu; aksi halde her 200 işte 30 rastgele çıktı (sınıf oranı korunarak), sapma > 0.1 ise yeniden.

**Kapasite (`eval/capacity.py`):** düzeltme süresi sağa çarpıktır, sınıflar arası fark büyüktür; ortalama yanıltır. Raporlanan: sınıf başına ve toplamda medyan ve p90 dakika; `iş/gün_tipik = desenci_saat × 60 / medyan`, `iş/gün_kötü = desenci_saat × 60 / p90`; `accepted` oranı sınıf başına.

---

## Lisans tablosu

| Bileşen | Lisans | Not |
|---|---|---|
| SAM 3.1 (facebookresearch/sam3) | SAM License | Ticari serbest, HF gated, dağıtımda lisans kopyası zorunlu, askeri/ITAR yasak. Erişim yoksa manuel köşe. |
| timm | Apache 2.0 | `segment` encoder (convnext_tiny) |
| segmentation-models-pytorch | MIT | yedek dekoder |
| opencv-contrib-python | Apache 2.0 | ximgproc, ECC, TPS |
| scikit-image, scipy, numpy, tifffile | BSD | |
| scikit-learn | BSD | k-means, çapraz doğrulama, kappa |
| statsmodels | BSD | QC ordinal regresyon |
| pyyaml | MIT | |
| noise | MIT | simplex gürültü |
| ultralytics | AGPL | KULLANMA |

---

## Girdi

| Alan | Kaynak | Zorunlu |
|---|---|---|
| Fotoğraf — uzun kenar ≥ `photo_long_px_min` (3000 px), önerilen ≥ 4000; JPEG kalite ≥ 90 | müşteri | evet |
| `width_cm`, `height_cm` — hav alanı, saçak hariç | müşteri | evet |
| `density` — anlamı `density_definition` ile | müşteri / fabrika | evet |
| `machine_colors` (5/6/8/12) | fabrika | evet |
| `consent_train: true/false` | müşteri | evet |
| `design_class` — sınıf adı; `auto` sadece sentetik/toplu değerlendirmede | desenci | **evet** (gerçek işte) |
| Tezgah profili, sınıf dosyaları | config | evet |
| İplik kütüphanesi | `config/yarns/*.csv` | hayır |
| `design_mode` (sınıf varsayılanını ezer) | desenci | hayır |
| `corners_override`, `palette_override`, `dither_pattern_override` | desenci | hayır |
| `force_symmetry`, `force_border_repeat`, `force_ground_repeat` (`auto|on|off`) | desenci | hayır |
| `regularize: auto|on|off`, `dither: auto|on|off` | desenci | hayır |
| `allow_low_res`, `allow_partial`, `no_gate` | desenci | hayır |

## Çıktı

```
runs/<job_id>/
├── output/
│   ├── design.tiff        indexed TIFF; one index per yarn
│   ├── palette.csv        index -> yarn_code (may be empty), name, RGB, knots, coverage %, quantization_deltaE
│   ├── preview.png        photo | rectified | model output | design (before/after regularize) | after dither | rerender
│   ├── qc.json
│   └── report.json
└── stages/<nn>_<name>/    intermediate outputs, cache; stages/guard.json
```

`report.json`:
```json
{
  "versions": {"git_sha": "...", "profile_hash": "...", "class_config_hash": "...", "yarn_csv_hash": "...",
               "sam_checkpoint": "facebook/sam3.1", "segment_ckpt_sha256": "...", "qc_calib_hash": "..."},
  "setup": {"reed_per_m": "profile", "density_definition": "knots_m2", "dither_pattern": "archive",
            "yarn_library": false, "motif_min_knots": "archive", "noise_scale": "assumed", "qc_calib": true},
  "photo": {"native_long_px": 4032, "below_min_px": false},
  "grid": {"W": 840, "H": 1140, "reed_per_m": 420, "wefts_per_m": 380, "grid_verified": true,
           "photo_px_per_knot": 2.7, "edge_width_knots": 1.3, "photo_px_per_knot_eff": 2.1,
           "knot_grid": {"visible": true, "spacing_px": [4.02, 3.97], "mismatch_pct": 0.6},
           "upsampled": true},
  "design_class": "gradient", "class_source": "designer",
  "class_auto": "gradient", "class_conf": 0.41, "class_conflict": false, "class_mixed": false,
  "class_features": {"edge_density": 0.04, "n_colors_est": 6, "gradient_frac": 0.38,
                     "noise_frac": 0.01, "noise_1knot_frac": 0.03, "line_frac": 0.2},
  "colors": {"machine_colors": 12, "palette_used": 9},
  "fringe": {"detected": [true, true, false, false], "strip_px": [42, 39, 0, 0]},
  "illumination": {"range_L": 11.4, "residual": 0.8, "flattened": true},
  "internal_geometry": {"source": ["knot_grid", "ground_repeat"], "tps_applied": true,
                        "max_dev_knots_before": 2.3, "max_dev_knots_after": 0.4},
  "palette_confirmed": false,
  "design_mode": "segment", "design_mode_requested": "segment", "design_mode_class_default": "segment",
  "regularize": {"applied": true, "changed_pct": 0.014, "line_jaggedness_before": 0.41,
                 "line_jaggedness_after": 0.12, "repeat_mismatch_raw": 0.06,
                 "segment_classes": {"line": 0.31, "arc": 0.12, "free": 0.57, "free_len_share": 0.62}},
  "guard": {"reverted_components": 2, "reverted_area_pct": 0.0003,
            "by_stage": {"production_cleanup": 0, "production_merge": 1, "regularize_repeat": 0, "regularize_smooth": 1}},
  "dither": {"applied": true, "mask_pct": 0.38, "levels": 4, "pairs": [[2, 5], [5, 7]]},
  "qc": {"status": "needs_review@4", "score": 0.61, "p5": 0.22, "target": 4,
         "features": {"rerender_edge_iou": 0.48, "banding": 1.1, "noise_1knot_pct": 0.02},
         "hard_gate_failures": [],
         "reasons": ["rerender_edge_iou=0.48", "palette_deltaE_p95=14.1", "banding=1.1"]},
  "timings_s": {"rectify": 3.1, "structure": 2.0, "design": 12.0, "regularize": 4.2, "dither": 0.6, "qc": 2.4, "total": 25.0},
  "warnings": [],
  "partial_view": false, "non_planar": false, "non_planar_internal": false,
  "deterministic": true
}
```

- `design_mode ≠ design_mode_class_default` → desenci ezmiş veya `segment` hata vermiştir; `warnings` nedenini yazar.
- `palette_confirmed`: `rerun --from production --palette ...` veya `--confirm-palette` ile `true`.

Saklama: `consent_train: false` → `retention_days` sonra silinir; `true` → `data/real/pairs/`.

### Determinizm

Kural: **eşitlikte en düşük indeks / en küçük mutlak eğim / en küçük tam sayı kazanır.**
- k-means: `random_state = 0`, `n_init` sabit, piksel sırası sabit.
- En yakın palet girişi (ΔE eşit): en düşük indeks.
- Mode pooling: en sık indeks; eşitlikte merkeze en yakın piksel; hâlâ eşitse en düşük indeks.
- Çoğunluk oyu (`regularize`): en düşük indeks.
- DP parçası açı kilidi: iki eğime eşit uzaklıkta → mutlak değeri küçük olan; eşitse pozitif.
- Çember fit ve doğru fit artıkları eşit → doğru.
- Yeniden rasterleştirme: bileşenler alan azalan, eşit alanda etiket artan sırada.
- Dither: ordered matris, pozisyona bağlı.
- `segment`: CPU/GPU'da tolerans içinde tekrarlanabilir (`test_seg.py`).

---

## Aşamalar

Sıra: `intake → rectify → structure → palette → design → production → regularize → dither → qc → export`. Her aşama tek girdi → tek çıktı; girdi hash'i (çözülmüş parametre seti dahil) değişmediyse yeniden çalışmaz. `rerun --from <ad>` o aşama ve sonrasını çalıştırır.

Çalışma çözünürlüğü tek ve sabittir: **4 px/düğüm**, `rectified.png` tam olarak `4W × 4H`, düğüm ızgarası piksel ızgarasına kilitli. İstisna yoktur.

### intake

Girdi: fotoğraf, iş parametreleri, profil, sınıf dosyaları, iplik CSV (opsiyonel)
1. Doğrulama: dosya; `width_cm`/`height_cm` > 0; `density` > 0; `machine_colors ∈ allowed_machine_colors`; `consent_train` verilmiş; `design_class` verilmiş ve `config/classes/<class>.yaml` var, ya da `auto` (`--class auto` sadece `synthetic`/`batch` bayrağıyla kabul edilir; gerçek işte hata). Override alanları doğrulanır.
2. EXIF döndürme. `photo.native_long_px`; `< photo_long_px_min` → `warnings += "photo_below_min_px"` (bilgi; asıl kapı `rectify` adım 12). Uzun kenar üst sınırı `max(4096, 4 × max(W, H))`.
3. Izgara. `reed_per_m` (yatay) ve `wefts_per_m` (dikey). `density_definition`:
   - `knots_m2`: `wefts_per_m = density / reed_per_m`.
   - `wefts_per_m`: `wefts_per_m = density`.
   - `reed_x_wefts`: `density = [tarak, atkı]`; `density[0] ≠ reed_per_m` → `grid_input_mismatch` hata.
   - `W = round(width_cm / 100 × reed_per_m)`, `H = round(height_cm / 100 × wefts_per_m)`; yuvarlama hatası > 0.5 düğüm → `grid_rounding`.
   - `reed_per_m: null`: sadece `knots_m2` ile, `k = sqrt(density)`; `grid_unverified`. `structure` gerçek tarağı ölçer (`grid.measured_reed_per_m`).
4. Palet üst sınırı `machine_colors`.
5. İplik kütüphanesi varsa CSV → `yarn_lab.npy`.
6. Parametre katmanı ilk çözüm (sınıf verilmişse nihai; `auto` ise profil değerleriyle, `structure` sonrası yeniden).
7. `versions`, `setup`, `photo` → `job.json`.
Çıktı: `job.json`, `photo.png`, `yarn_lab.npy` (opsiyonel)

### rectify

Girdi: `photo.png`, `job.json`
1. Köşe kaynağı: `corners_override` varsa maske = dörtgen, adım 4'e. Yoksa SAM 3.1, prompt `"carpet"` → `"rug"` → `"carpet on the floor"` (skor < 0.3'te sıradaki), seçim `score × alan`. SAM yoksa `sam_unavailable`, `needs_corners` ile dur.
2. En büyük bağlı bileşen, delik doldurma.
3. Saçak tespiti: kaba homografi ile düzleştirilmiş maskede 4 kenar için 1-B profil (doygunluk ortalaması, 5 px yerel varyans). Düşük doygunluk + yüksek varyans şerit, ardından keskin geçiş (CUSUM). Genişlik kenarın %2–`fringe_max_frac`'ı arasında ve kısa kenarlardan en az birinde varsa kabul, maske geçişe kırpılır. Belirsizse `fringe_uncertain`. Uzun kenarda `fringe_long_side`.
4. Kısmi kadraj: maske görüntü kenarına 3 px'ten yakın ve temas çevrenin > %10'u → `partial_view`. `allow_partial: false` ise dur.
5. Köşe: `approxPolyDP` (eps = çevre × 0.02) → 4 köşe; olmazsa `HoughLinesP` kesişimleri; olmazsa `minAreaRect` + `corners_weak`.
6. Düzlemsellik (dış): kontur–dörtgen en büyük dik uzaklık / kenar → `planarity_dev`; `> planarity_warn` → `non_planar`; `> planarity_stop` → dur.
7. Geometrik çözünürlük: `photo_px_per_knot = min(4 kenar için kaynak px / düğüm)`. `upsampled = photo_px_per_knot < 4`.
8. Homografi: 4 köşe → `(0,0), (4W,0), (4W,4H), (0,4H)`. INTER_AREA küçültme / INTER_CUBIC büyütme.
9. Gray-world beyaz dengesi → `wb_gains`, `wb_clipped_pct`.
10. Aydınlatma düzleştirme: Lab L, log uzayı; maske = guided filter + Canny kenarlarının 3 px dilate'i dışı; 2. derece iki değişkenli polinom, Huber. Ortalama L korunur. `illum_range > illum_warn` → `strong_illumination`. a/b düzleştirilmez.
11. `rectified.png` (renk-doğru) ve `rectified_contrast.png` (L CLAHE clip 2.0 tile 8; sadece `structure` ve kenar işlemleri).
12. Kenar keskinliği: Canny kenar noktalarında gradyan yönünde %10–%90 geçiş genişliği, medyan / 4 → `edge_width_knots`. `photo_px_per_knot_eff = photo_px_per_knot / max(1, edge_width_knots)`. `< photo_px_warn` → `low_photo_resolution`; `< photo_px_stop` → dur (`allow_low_res` → `forced_low_res`). `edge_width_knots > pile_blur_knots` → `pile_blur`.
Çıktı: `rectified.png`, `rectified_contrast.png`, `mask.png`, `corners.json`, `job.json` güncellenir
Hata: maske alanı < %20 → `needs_corners`.

### structure

Girdi: `rectified_contrast.png` (INTER_AREA ile 512 px; adım 5b, 7, 8, 9 tam çözünürlük), `rectified.png`, `job.json`
1. Yatay/dikey yansıma korelasyonu → simetri none / h / v / hv, güven = korelasyon.
2. Bordür: kenardan içeri satır/sütun varyansı → kalınlıklar ve iç kenar konumları. Kabul: dört kenarda tespit **ve** adım 3'te `border_period_conf` dolu **ve** kalınlık / kısa kenar ∈ `[border_frac_min, border_frac_max]`. Aksi halde `border: null`, `frame_hint: panel`, şerit zemin sayılır.
3. Bordür şeridinde 1-B otokorelasyon → `border_period_px`, güven (< 0.6 → null). Üst/alt ve sol/sağ ayrı; köşe blokları otokorelasyona girmez.
4. Merkez %40'ta kapalı yüksek kontrast bölge → madalyon.
5. Zemin/madalyon 2-B tekrar: 2-B otokorelasyon, en güçlü iki bağımsız tepe → `(v1, v2)`, güven (< 0.6 → null). Kafes düğüm noktaları `lattice_nodes`.
   5b. Gradyan maskesi (tam çözünürlük): 32×32 düğüm bloklarında `edge_density` düşük **ve** düşük frekanslı (σ = 8 düğüm) tek yönlü rampa (gradyan büyüklüğü ∈ `[grad_min, grad_max]`, yön tutarlılığı ≥ 0.8) → bloklar birleştirilir, 8 düğüm morfolojik kapama, alanı < 64 düğüm atılır → `gradient_mask.npy`, `gradient_regions.json` (rampa yönü, Lab uç değerleri). `gradient_frac` = maske alan oranı.
6. Kaba etiket haritası ve sınıf tespiti:
   - 400 px Lab k-means (k = 16, sabit seed), merkezler ΔE < 8 ise birleştirilir → `n_colors_est` ve merkezler.
   - `rectified` INTER_AREA ile **W×H'ye** (ızgara çözünürlüğü); her düğüm en yakın merkeze atanır (yeniden fit yok) → `coarse_labels.npy` (H×W). `noise_frac`, `noise_1knot_frac` ve adım 9'un bileşenleri bu haritadan.
   - Sınıf tespiti (bkz. Desen sınıfı). `class_source: designer` ise sadece rapor + `class_conflict`. `class_mixed` her durumda değerlendirilir.
   - `auto` ise parametre katmanı yeniden çözülür.
7. İç geometri ölçümü, düğüm cinsinden sapma, üç kaynak:
   - `border`: iç kenar dizisi doğrusal regresyona uydurulur, maksimum sapma.
   - `ground_repeat`: `lattice_nodes` ile ideal kafes arası sapma.
   - `knot_grid`: `photo_px_per_knot_eff ≥ 2` ve `pile_blur` yoksa `rectified_contrast` tam çözünürlükte 64×64 px bloklarda 2-B otokorelasyon; beklenen tepeler (4,0), (0,4) px; tepe ≥ 0.4 güvenli blok oranı ≥ 0.7 → `visible`. (a) Global medyan aralık; `|spacing − 4| / 4 > grid_mismatch_tol` → `grid_mismatch` (sert kapı). Gerçek tarak → `measured_reed_per_m`. (b) Blok bazında sapmadan yer değiştirme alanı → `internal_dev_knots`, TPS kontrol noktaları.
   `internal_dev_knots` = maksimum. `> planarity_warn × kenar` → `non_planar_internal`. Üç kaynak da yoksa `internal_geometry: unmeasurable` (sert kapı).
8. TPS düzeltmesi (`non_planar_internal` ise): kontrol noktaları bordür iç kenarlarında 16'şar nokta, `lattice_nodes`, `knot_grid` alanından 8×8 örnek, simetri ekseninde 8 nokta, dış 4 köşe sabit. `cv2.createThinPlateSplineShapeTransformer` ile `rectified*.png` warp (4W × 4H korunur). Doğrulama: adım 7 tekrar; `internal_dev_knots` ≥ %50 düşmediyse geri alınır, `tps_no_gain`. Düştüyse `tps_applied: true`.
9. Bölge haritası, **ızgara çözünürlüğünde**: 512 px'te tespit edilen bölge maskesi (zemin/bordür/madalyon/köşe) ızgaraya yukarı örneklenir; `coarse_labels.npy` (adım 6, zaten W×H) üzerindeki bağlı bileşenler **bütün olarak** en çok örtüştüğü bölgeye atanır → `regions.npy` (H×W, uint8): 0 = ground, 1 = border, 2 = medallion, 3 = other, 4 = corner. Gerekçe: bileşen sınırları ızgara çözünürlüğünde hesaplanmazsa (400 px'te olsaydı) aşamanın amacı (±2–3 düğüm sınır hatasını gidermek) bozulur. Tek tüketici `regularize`.
Çıktı: `layout.json`, `regions.npy`, `coarse_labels.npy`, `gradient_mask.npy`, `gradient_regions.json`, güncellenmiş `rectified*.png` (TPS uygulandıysa), `job.json` (`resolved_params`)

Zorlama kararı (`regularize`'da): `force_*` — `auto` → ilgili güven ≥ `enforce_conf` **ve** `design_class ∈ {classic, geometric}`; `on` → her durumda (tespit yoksa uyarı ile atlanır); `off` → asla. Zorlama guard'dan geçer. Tespit eşiği (0.6) ile zorlama eşiği (0.85) ayrı.

### palette

Girdi: `rectified.png`, `gradient_mask.npy`, `gradient_regions.json`, `job.json`, `yarn_lab.npy` (opsiyonel), `palette_override`
Palet modelden **önce**, fotoğraftan seçilir. Üst sınır `machine_colors`, alt sınır 1.
1. `rectified` → guided filter (r=8, eps=0.01) → Lab → uzun kenar 400 px → k-means (k = 4 × `machine_colors`, sabit seed). Küme ağırlığı `count^area_weight_exp`. Gradyan maskesi içindeki pikseller ağırlık 0; her gradyan bölgesinin iki uç Lab değeri zorunlu aday.
2. Komşuluk matrisi: 400 px'te iki kümenin paylaştığı sınır piksel sayısı; gradyan maskesi içindeki sınırlar sayılmaz.
3. Seçim:
   - `palette_override` varsa: verilen iplikler.
   - Kütüphane varsa: aday = tüm iplikler; açgözlü seçim, her adımda ağırlıklı toplam ΔE2000'i en çok düşüren iplik; kazanç < `gain_stop_select` (ΔE birimi) → dur. Gradyan uçları için en yakın iplik zorunlu.
   - Kütüphane yoksa: açgözlü küme birleştirme, maliyet = ağırlıklı ΔE artışı + `adjacency_penalty × paylaşılan sınır / toplam sınır`. `machine_colors`'a in; ardından birleştirme maliyeti `merge_cost_stop` altında kaldığı sürece birleştirmeye devam et. Gradyan uç kümeleri birleştirilmez.
   - İki durdurma eşiği ayrıdır: biri kazanç (ΔE, pozitif yönde), diğeri maliyet (ΔE + ceza); aynı sayıyı paylaşamazlar.
4. Motif rengi koruması: kapsama küçük ama bileşen sayısı ≥ `motif_color_min_cc` veya en büyük bileşeni ≥ `motif_min_knots` olan kümeler birleştirmede son sırada.
5. `palette.json` (indeks, RGB, Lab, `yarn_code`, `gradient_endpoint`), `palette_proposed.json`.
Çıktı: `palette.json`, `palette_proposed.json`

### design

Girdi: `rectified.png`, `gradient_mask.npy`, `palette.json`, `job.json`, çözülmüş parametreler
Simetri/tekrar zorlama burada yapılmaz (`regularize`).

**segment (classic, geometric, gradient, minimal varsayılanı)**
Deterministik. Motif uydurmaz; girdi belirsizse bulanık çıktı verir, `production` en yakın ipliğe kuantize eder, hata görünür kalır.
1. Model `SegNet` (`design_seg.py`): `timm.create_model("convnext_tiny", features_only=True, out_indices=(0,1,2,3))` + FPN dekoder, stride 4 → düğüm başına 4 kanal: Lab (3, regresyon) + sınır olasılığı (1, sigmoid). Yedek: `smp.Unet(convnext_tiny)` + `avg_pool2d(4)`.
2. Karo çıkarımı: 512×512 px (128×128 düğüm), 4 px katı, örtüşme 64 px, merkez 448×448 alınır.
3. Gradyan bölgede istenen davranış: düz (yerel ortalama) Lab + düşük sınır olasılığı; eğitim hedefi buna göre kurulur. Ara renkleri `dither` üretir.
4. Model dosyası yoksa veya çıkarım hata verirse `design_mode → quantize`, `warnings += "segment_fallback"`.
Çıktı: `design_lab.npy` (H×W×3), `boundary.npy` (H×W), `design_rgb.png`

**quantize (taban çizgisi, `distressed` varsayılanı, yedek)**
1. `rectified.png` 4 px/düğüm'de guided filter (r=8, eps=0.01), mean-shift (sp=24, sr=20).
Çıktı: `design_rgb.png`
`distressed` gerekçesi: mean-shift gürültüyü olduğu gibi bırakır; `segment`'in sınır kanalı eskitme gürültüsünde doyar ve L1(Lab) gürültünün ortalamasını öğrenir. `segment`'in `distressed` eval'i (`noise_fidelity`) bunu değiştirirse sınıf dosyası güncellenir.

### production

Girdi: `design_rgb.png` veya `design_lab.npy` + `boundary.npy`, `palette.json`, `gradient_mask.npy`, `job.json`, çözülmüş parametreler
1. Izgara W×H.
2. Kuantizasyon:
   - `quantize`: her piksel → en yakın giriş (ΔE2000); 4×4 mode pooling.
   - `segment`: iki geçiş. Geçiş 1: `boundary ≤ boundary_thr` → en yakın giriş. Geçiş 2: sınır düğümleri → `boundary_snap_r` yarıçapındaki geçiş-1 girişlerinden tahmin Lab'ına en yakını. Gradyan maskesinde iki geçiş uygulanmaz: tahmin Lab'ı `design_lab_smooth.npy` (dither girdisi), indeks geçici olarak en yakın giriş.
3. Gürültü ölçümü (temizleme öncesi): `noise_knots_pct` (alanı ≤ `min_region_knots`), `noise_1knot_pct` (alanı = 1), gradyan maskesi dışı.
4. Bölge temizleme (3 tur): alanı ≤ `min_region_knots` bileşen → en uzun sınırı paylaştığı komşuya. `segment`'te `boundary > 0.5` düğümler muaf; gradyan maskesi muaf; `min_region_knots: 0` → atlanır. `cleanup_changed_pct`. **Guard.**
5. `is_thin(bileşen)`: iskelet uzunluğu ≥ `line_min_knots` VE maksimum yerel genişlik ≤ `line_max_width`.
6. Kapsama < `min_coverage_pct` renkleri en yakın girişe birleştir. Muaf: `is_thin` bileşenlerden oluşan renkler; motif rengi kriteri; `gradient_endpoint`. `min_coverage_pct: 0` → atlanır. **Guard.**
7. Rapor: giriş başına `quantization_deltaE`; `palette_deltaE` p50/p95; `palette_model_mismatch` (p95 > `palette_model_mismatch_warn`).
Çıktı: `indexed_raw.npy`, `design_lab_smooth.npy` (gradyan varsa), `production_stats.json`, `stages/guard.json` (eklenir)

### regularize

Girdi: `indexed_raw.npy`, `layout.json`, `regions.npy`, `gradient_mask.npy`, çözülmüş parametreler
Deterministik; model yok. `regularize: off` → `indexed = indexed_raw`, istatistikler yine hesaplanır. Gradyan maskesi ve `regions == 4` tüm adımlarda dokunulmaz.
1. Ölçüm (öncesi): `line_jaggedness_before`, `repeat_mismatch_raw`, `symmetry_mismatch_raw` (yoksa null).
2. Tekrar ve simetri birleştirme — **şablon önce düzleştirilir, sonra basılır**: tespit → çoğunluk oyu ile şablon → şablona adım 3 → N kere basılır. Kopyalar tanım gereği tutarlı.
   - Bordür (1-B periyot): dört kenar şeridi ayrı; üst/alt ve sol/sağ ayrı şablon. Periyot tam sayı değilse en yakın tam sayı, kalan fark köşe bloğuna bitişik uçta. Köşe blokları muaf.
   - Zemin/madalyon (2-B): kafes `(v1, v2)` üzerinde çoğunluk oyu → tek hücre şablonu. Hücre sınırını kesen bileşenler için şablon 3×3 döşenir, adım 3 döşenmişte çalışır, merkez alınır. Kafes kenarındaki eksik hücreler şablonla tamamlanır.
   - Simetri: h/v/hv ayna konumları arası çoğunluk oyu → çeyrek/yarım şablon → adım 3 → aynalanarak basılır. Eksen düğüm üzerinde değilse eksen sütunu/satırı korunur.
   - Her yapı sonrası **guard**.
3. Sınır düzleştirme (adım 2'nin şablonları ve tekrar/simetri dışı bileşenler; alan büyükten küçüğe):
   - Sınır polilinesi; Douglas-Peucker, tolerans `reg_tol_knots` (0.7).
   - Parça sınıfı: **doğru** — eğim `{0, 90, ±1:1, ±1:2, ±2:1, ±1:3, ±3:1}` setine kilitlenir, parça < `reg_min_line_knots` ise kilitlenmez. **Yay** — çember uydurma, artık ≤ `reg_tol_knots` ve parça ≥ 8 düğüm. **Serbest** — kalanlar.
   - Serbest parçalar (`reg_free_runs`): rasterleştirilmiş basamak run-length dizisi 5'lik medyan filtre ile monoton parçalı sabit hale getirilir; toplam uzunluk ve uç noktalar korunur; değişim `reg_tol_knots` içinde.
   - Yeniden rasterleştirme (`skimage.draw.polygon`); büyükler önce.
   - Kapsama kuralı: hiçbir bileşene düşmeyen düğümler `indexed_raw` indeksini korur; ardından tek tur `min_region_knots` temizliği.
   - `is_thin` bileşenlerde iskelet bağlantısı kontrol edilir; koptuysa geri alınır.
   - **Guard.**
4. Ölçüm (sonrası): `line_jaggedness_after`, `repeat_mismatch` (sıfır olmalı), `symmetry_mismatch`, `regularize_changed_pct`.
5. `regularize_stats.json`: parça sınıf dağılımı (`segment_classes`, `free_len_share`).
Çıktı: `indexed.npy`, `regularize_stats.json`, `stages/guard.json` (eklenir)

### dither

Girdi: `indexed.npy`, `design_lab_smooth.npy`, `gradient_mask.npy`, `gradient_regions.json`, `palette.json`, `dither_pattern`
Gradyan maskesi yoksa boş geçer (`dither.applied: false`). Deterministik, analitik.
1. Her gradyan bölgesi için rampa: iki uç indeks (`gradient_endpoint`); arada Lab'da sıralı ara indeksler varsa çok segmentli `[i0, i1, ..., ik]`.
2. Her düğümün `design_lab_smooth` değeri rampaya izdüşürülür → komşu iki indeks `(a, b)`, oran `t`.
3. `t` `levels` kademeye yuvarlanır; ordered matris (`config/dither/<name>.yaml`: `type: ordered | row_scan | diagonal_scan`, `matrix`, `levels`) ile `a` veya `b`.
4. Maske kenarındaki düğümler en yakın uç indekse (dither düz alana taşmaz).
5. Rapor: `mask_pct`, `pairs`, `levels`. `dither_pattern: null` → `dither_pattern_default` (sert kapı).
Çıktı: `indexed.npy`

### qc

Girdi: `indexed.npy`, `indexed_raw.npy`, `palette.json`, `rectified.png`, `layout.json`, `gradient_mask.npy`, `production_stats.json`, `regularize_stats.json`, `stages/guard.json`, `job.json`, `models/qc_calib.json`
1. Sert kapılar.
2. Geri-render: `indexed` → palet RGB → 2 px/düğüm; sentetik zincirin hav dokusu (aydınlatma, perspektif, kamera yok) → `rerender.png`. `rectified` aynı boyuta.
3. Özellikler; null kuralları.
4. `qc_calib.json` varsa ve hash uyuşuyorsa skor + `p5` + karar + `reasons`; `class_unreliable` → `needs_review`; yoksa `needs_review`, `score = null`. Hash uyuşmazlığı → `qc_calib_stale`.
5. `preview.png` panelleri.
Çıktı: `qc.json`, `rerender.png`. Süre < 5 s.

### export

1. `design.tiff`: `tifffile`, `photometric='palette'`, colormap = palet RGB (uint16), `resolution` düğüm/inç (yatay = reed, dikey = atkı).
2. `palette.csv`: `index, yarn_code, name, R, G, B, knots, coverage_pct, quantization_deltaE`.
3. `preview.png`, `report.json`.
4. `consent_train: true` → `data/real/pairs/<job_id>/{photo.jpg, design.tiff, indexed_raw.npy, corners.json, meta.json}`, `reviewed=false`. Desenci nihai TIFF'i → `truth.tiff`, `reviewed=true`. QC özellikleri, puan, süre, tür, sınıf → `meta.json`.
Çıktı: `output/`

---

## Tezgah profili

```yaml
# config/looms/wilton_default.yaml
name: wilton_default
allowed_machine_colors: [5, 6, 8, 12]
reed_per_m: null              # warp ends per meter; loom constant. null -> square assumption + grid_unverified
density_definition: knots_m2  # knots_m2 | wefts_per_m | reed_x_wefts
photo_long_px_min: 3000       # customer-facing minimum; below -> photo_below_min_px warning (real gate is photo_px_per_knot_eff)
min_region_knots: 2           # components at or below this area are merged into a neighbor; 0 disables
line_min_knots: 6
line_max_width: 1
motif_min_knots: 10           # overwritten by tools/motif_stats.py
motif_min_knots_source: assumed
motif_color_min_cc: 5
noise_cc_min: 2               # noise_frac counts components with area in [noise_cc_min, noise_cc_max]
noise_cc_max: 6               # overwritten by tools/noise_stats.py
noise_scale_source: assumed
min_coverage_pct: 0.5
area_weight_exp: 0.5
adjacency_penalty: 2.0        # palette; ignored inside the gradient mask
gain_stop_select: 0.3         # library path: stop when weighted deltaE gain falls below this
merge_cost_stop: 0.5          # no-library path: keep merging while merge cost stays below this (initial guess)
guard_warn_pct: 0.02
deltaE_warn: 3.0              # placeholder; fixed after designer measurement
palette_model_mismatch_warn: 12.0
piece_rug: true
allow_partial: false
allow_low_res: false
fringe_detect: true
fringe_max_frac: 0.15
planarity_warn: 0.02
planarity_stop: 0.05
border_frac_min: 0.03
border_frac_max: 0.30
illum_poly_deg: 2
illum_warn: 15.0
photo_px_warn: 1.5            # applied to photo_px_per_knot_eff
photo_px_stop: 0.8
pile_blur_knots: 2.0
grid_mismatch_tol: 0.05
edge_density_min: 0.02
gradient_class_frac: 0.25
noise_class_frac: 0.08
class_conf_min: 0.2
design_mode: segment          # segment | quantize
regularize: auto
reg_tol_knots: 0.7
reg_min_line_knots: 4
reg_free_runs: true
enforce_conf: 0.85
force_symmetry: auto          # auto only enforces for classic/geometric
force_border_repeat: auto
force_ground_repeat: auto
dither: auto                  # auto = on when a gradient mask exists
dither_pattern: null          # null -> 4x4 ordered + dither_pattern_default warning
dither_levels: 4
boundary_thr: 0.5
boundary_snap_r: 2
qc_accept: 0.8                # placeholder until models/qc_calib.json exists
qc_reject: 0.3
retention_days: 30
```

Gradyan maskesi içindeki muafiyetler (`adjacency_penalty: 0`, `min_region_knots: 0`, `min_coverage_pct: 0`, `regularize: off`) sınıf dosyasında değil, ilgili aşamaların maske koşullarında sabittir; her sınıfta geçerlidir.

### Sınıf dosyaları

```yaml
# config/classes/gradient.yaml
qc_features_extra: [banding]
```
```yaml
# config/classes/distressed.yaml
design_mode: quantize
regularize: off
min_region_knots: 1           # remove single-knot islands only; set to 0 if noise_stats.py finds intentional 1-knot noise in the archive
min_coverage_pct: 0.0
force_symmetry: off
force_border_repeat: off
force_ground_repeat: off
qc_features_null: [noise_knots_pct, cleanup_changed_pct]
```
```yaml
# config/classes/minimal.yaml
force_symmetry: off
```
`classic.yaml` ve `geometric.yaml` boştur.

## İplik kütüphanesi (opsiyonel)

```
yarn_code,name,R,G,B,material
12,Beige,204,183,152,Wool
```

---

## Veri (data/)

```
data/
├── real/
│   ├── pairs/<id>/{photo.jpg, truth.tiff, corners.json, H_refined.json, H_local.json,
│   │              align_mask.npy, gradient_mask.npy, noise_mask.npy, meta.json}
│   └── splits.json
├── archive/designs/<id>/{design.tiff, palette.csv}
├── backgrounds/*.jpg
└── synth/
    ├── dist.yaml              # real-world distributions (colors, size, density, px/knot); defaults until real pairs exist
    ├── pairs/<id>/{photo.jpg, truth.tiff, meta.json, corners_gt.json, layout_gt.json}
    ├── generate.py            # design -> photo (render chain)
    └── generate_designs.py    # procedural designs with freely combined class features
```

`meta.json` (gerçek çift): `consent_train`, `reviewed`, `design_class` (desenci), `width_cm`, `height_cm`, `reed_per_m`, `wefts_per_m`, `machine_colors`, `photo_native_px`, `qc` özellikleri, `qc_status`, `designer_score`, `fix_minutes`, `fix_type`.

### Gerçek çift toplama

Kritik eşikler (sınıf başına sayılır; "ne varsa al" dağılımı satılmış desenlerin dağılımı yapar):
- `test_core_modern` gerçek sürümü için ≥ 20 modern test çifti (`geometric | gradient | distressed | minimal`).
- QC kalibrasyonu için sınıf başına ≥ 10 (altı `other`'a düşer, o sınıfta karar verilmez).
- SegNet için hizalamayı geçen ≥ 40 çift (`min_real_segment`).

Toplama anında kaydedilmesi zorunlu olanlar:
- `design_class` (desenci söyler).
- `width_cm`, `height_cm`, yoğunluk, `machine_colors`.
- Fotoğraf **o desenin üretilmiş halısının** olmalı; benzerinin değil (aksi halde ECC hizalama tutmaz).
- `truth.tiff` nihai üretim dosyası olmalı; ara taslak değil.
- Fotoğraf uzun kenarı ≥ `photo_long_px_min`; altındakiler `low_photo_resolution`'a takılır ve boşa gider.

Son üç madde en çok çift kaybettiren yerdir; `tools/pair_intake.py` bu alanları toplama anında doğrular, eksik olanı reddeder.

`train/align.py` ilk 20–30 çift gelince çalıştırılır, 200 çift sonra değil. Elenme oranı yüksekse toplama kriteri düzeltilir.

`tools/pair_stats.py` gerçek çiftlerden renk sayısı, ebat, tarak/atkı ve `photo_px_per_knot` dağılımlarını çıkarır → `data/synth/dist.yaml`. Sentetik üreteçler bu dosyadan örnekler; gerçek çift yokken varsayılan aralıklar (renk 2–12, ebat 60×90 – 300×400 cm, tarak 300–500, px/düğüm 0.8–4) kullanılır ve `dist.yaml.source: default` yazılır.

### Bölünme

`test` ve `val` sadece gerçek; `train` = kalan gerçek + tüm sentetik. `splits.json` git'te kilitli, `design_class` ile katmanlı. Aynı halının farklı fotoğrafları aynı bölümde. İki sabit çekirdek set:
- `test_core_classic`: ilk 20 `classic` test id.
- `test_core_modern`: ilk 20 modern test id. 20 gerçek modern çift birikene kadar 100 sabit prosedürel sentetik (`eval/synth_modern_ids.txt`); gerçek set dolunca ikisi de raporlanır.

### Gerçek çift hizalama (`train/prepare.py` → `train/align.py`)

1. `truth.tiff` → palet RGB → `rectified` boyutuna INTER_NEAREST.
2. Global ECC (homografi, Lab L, 3 seviye piramit) → `H_refined.json`.
3. Yerel ECC: 4×4 blok, blok başına affine → `H_local.json`; blok `align_px` ve `edge_fidelity`.
4. `align_mask.npy`: blok `align_px ≤ 4` VE fidelity ≥ 0.3 → 1. `align_px > 2` bloklarda truth sınırlarının 1 düğüm komşuluğu maskeden çıkarılır.
5. Truth'ta dither bölgeleri (`dither_stats.py` kuralı) → `gradient_mask.npy`; truth'ta gürültü bölgeleri (`noise_stats.py` kuralı: 16×16 düğüm penceresinde alanı ≤ `noise_cc_max` bileşen yoğunluğu ≥ eşik) → `noise_mask.npy`.
6. Çift düzeyi: global `align_px` medyanı büyük ve blokların > %50'si maske 0 → `layout_changed`, eğitimden düş.
7. `align_report.csv`: çift başına `align_px`, maske doluluk %, `design_class`, karar; sınıf kırılımlı özet.

Düşme politikası: kalan ≥ 40 ve doluluk ≥ %60 → normal; < 40 → blok eşiği 6 px; hâlâ < 40 → `segment` sentetikle eğitilir, gerçek çiftler val/test'te.

### Sentetik çift üreteci (`data/synth/generate.py`)

Zincir: desen render → hav dokusu (ızgaraya kilitli) → iplik sapması → lif bulanıklığı (`blur_knots`) → hav yönü gölgelenmesi → aydınlatma → perspektif → arka plan → okluzyon → kamera → JPEG (kalite 75–95).
- Aynı RGB'li indeksler render öncesi birleştirilir; `truth.tiff` birleştirilmiş halidir.
- Hav yönü gölgelenmesi `m(x,y) = 1 + a·(d·p) + b·kabarma(p)`, `a ∈ [0.05, 0.25]`, `b ∈ [0, 0.1]`.
- `blur_knots ∈ [0.3, 3.5]`.
- Perspektifte %15 düzlem dışı TPS bozulma.
- Saçak %30.
- Kamera: `photo_px_per_knot` `dist.yaml`'dan.
- Kamera gürültüsü ve JPEG artefaktı tek düğüm ölçeğinde fotoğraf gürültüsü üretir; `noise_1knot_frac` ve sınıf tespitindeki tek düğüm dışlaması bununla test edilir.
- `layout_gt.json`: bordür, periyot, köşe blokları, tekrar vektörleri, simetri, gradyan maskesi, gürültü maskesi, `design_class`, tüm sınıf oranları.

### Prosedürel desen üreteci (`data/synth/generate_designs.py`)

Klasik motif için arşiv gerekir; diğer dört sınıf arşivsiz üretilir. Sınıf özellikleri **serbestçe kombine edilir**: her desen bir taban şablondan başlar (`geometric | gradient | distressed | minimal`, kota ile) ve bağımsız olasılıklarla bordür, kafes tekrar, gradyan bölgesi (1–3), simplex gürültü (`noise_frac ∈ [0.05, 0.25]`), simetri alır. Meta'ya `gradient_frac`, `noise_frac`, `has_border`, `has_repeat`, `symmetry`, `n_colors` yazılır.
- Taban şablon dışında hiçbir özellik almayan desenler `pure: true`; `class_acc` bunlarda ölçülür.
- `gradient_frac ≥ 0.25` ve `noise_frac ≥ 0.08` olanlar `mixed: true`; `class_mixed` uyarısının tetiklenme oranı bunlarda ölçülür.
- Renk sayısı, ebat, yoğunluk `dist.yaml`'dan.
- 100 sabit id `eval/synth_modern_ids.txt` (pure ve mixed karışık, oranlar dosyada).

---

## Eğitim (train/)

### Hazırlık (`train/prepare.py`)
1. Her çift `intake`–`structure`'dan geçer (gerçek çiftte manuel köşe, sınıf desenciden).
2. Hedefler (`segment`):
   - `target_lab.npy`: truth palet Lab; gradyan maskesinde (truth dither ∪ fotoğraf gradyan maskesi) 4×4 düğüm kutu ortalaması.
   - `target_boundary.npy`: indeks değişimi olan düğümler, 1 düğüm dilate; gradyan maskesinde 0 (kayıp maskelenmez; model gradyan bölgede "sınır yok + düz Lab" üretmeyi açıkça öğrenir).
   - `loss_mask.npy = align_mask ∧ ¬noise_mask`.
   - **Atma birimi karo, çift değil.** `distressed` çiftler eğitime girer; `noise_mask` kapsaması > `tile_skip_noise_frac` (0.2) olan karolar atlanır, kalan karolar `loss_mask` ile kullanılır. Gürültü bölgesi öğrenilmez, temiz bölge kaybedilmez.
3. Augmentasyon: çevirme (çift birlikte), döndürme yok, renk jitter sadece girdi (±%10), çevrimiçi hav gölgelenmesi, lif bulanıklığı (0.3–3 düğüm), kamera gürültüsü, JPEG.
4. `train/data/{train,val}.jsonl` (`design_class`, karo listesi).

### SegNet eğitimi (`train/train_segment.py`)
- `SegNet` (convnext_tiny features_only + FPN, stride 4, 4 kanal). Encoder ImageNet.
- Girdi 4 px/düğüm, 512×512 karo, 4 px hizalı, batch 16, AdamW 3e-4, cosine, 40 epoch, bf16. Karo örnekleme `design_class` dengeli.
- Kayıp: `loss_mask` maskeli L1(Lab) + λ·BCE(sınır), λ = 1.0.
- Augmentasyon: çevirme, renk jitter, ölçek 0.9–1.1 (ızgara hizalı), çevrimiçi aydınlatma/gölgelenme/bulanıklık/gürültü, çözünürlük augmentasyonu (1–4 px/düğüm'e küçült, 4'e büyüt).
- Her 2 epoch'ta val: kayıp + sınıf başına 2 örnekte tam pipeline → eval metrikleri.
- Çıktı: `models/knotvision_segment.pt`.

---

## Değerlendirme (eval/)

### Offline (`eval/run.py`)

`test_core_classic` + `test_core_modern` (+ `test_full`) + 100 sentetik (arşiv) + 100 sentetik (prosedürel), her mod, `regularize` açık/kapalı. **Sonuç tablosu `design_class` kırılımıyla**; toplam ortalama ek sütun.

| Metrik | Tanım | Ölçtüğü şey |
|---|---|---|
| `corner_err_px` | sentetik: 4 köşe ortalama hata | `rectify` |
| `fringe_err_px` | sentetik saçaklı: hav kenarı hatası | saçak |
| `planarity_detect` | sentetik `non_planar` / `non_planar_internal` eşleşmesi | düzlemsellik |
| `tps_gain` | `internal_dev_knots` öncesi/sonrası | TPS |
| `grid_field_err` | `knot_grid` yer değiştirme alanı RMSE; `visible` doğruluğu; `measured_reed_per_m` hatası | doku tabanlı geometri |
| `illum_err` | kestirilen aydınlatma ile `meta.illum` L RMSE | düzleştirme |
| `edge_width_err` | `edge_width_knots − meta.blur_knots` | keskinlik ölçümü |
| `layout_err` | bordür, köşe bloğu, periyot, tekrar vektörü, simetri, gradyan maskesi IoU, gürültü maskesi IoU | `structure` |
| `class_acc` | `pure` sentetik ve etiketli gerçek: tespit doğruluğu, karışıklık matrisi; `class_uncertain` oranı | sınıf tespiti |
| `class_mixed_recall` | `mixed` sentetik: `class_mixed` tetiklenme oranı; `pure`'da yanlış tetiklenme oranı | karışık halı uyarısı |
| `class_conflict_rate` | gerçek: desenci sınıfı ile otomatik sınıf uyuşmazlığı, sınıf kırılımlı | tespit kalitesi |
| `knot_deltaE_mean`, `_p95` | düğüm başına çıktı − truth ΔE2000; gradyan maskesinde 8×8 blok | renk |
| `banding` | gradyan maskesinde düşük frekanslı L farkı RMS | bantlaşma |
| `dither_err` | 4×4 blok karışım oranı farkı | dither |
| `noise_fidelity` | truth `noise_mask` içinde: (a) düğüm bazlı indeks uyuşma oranı `noise_index_acc`, (b) çıktı / truth küçük bileşen sayısı oranı `noise_cc_ratio` (1'e yakın iyi; < 1 gürültü silinmiş, > 1 fotoğraf gürültüsü yazılmış). `noise_fidelity = noise_index_acc × min(r, 1/r)`. Maske dışında `noise_1knot_frac(çıktı) − noise_1knot_frac(truth)`. | gürültü sadakati; `distressed`'ın tek gerçek kalite ölçüsü |
| `thin_line_recall` | truth `is_thin` bileşenlerinin korunma oranı | ince çizgi |
| `cc_added_area_pct` | eşleşmeyen çıktı bileşenlerinin düğüm oranı (gradyan ve gürültü maskesi dışı) | uydurma (alan) |
| `cc_added_motif` | eşleşmeyen, alanı > `motif_min_knots`, `is_thin` olmayan çıktı bileşeni sayısı | uydurma motif (sert, 0) |
| `cc_missing_pct` | eşleşmeyen truth bileşeni oranı; aşama kırılımı `guard.json`'dan | eksik motif |
| `edge_iou` | 1 düğüm toleransla sınır IoU | konum |
| `line_jaggedness_delta` | çıktı − truth | düzenlilik |
| `free_len_share` | serbest parça uzunluk payı | vektörleştirme yeterliliği |
| `repeat_mismatch`, `symmetry_diff` | | tekrar, simetri |
| `regularize_changed_pct`, `guard_reverted_pct` | | yük |
| `qc_score`, `qc_status`, `qc_p5` | QC çıktısı; `knot_deltaE` ve desenci puanı ile korelasyon, sınıf başına | QC uyumu |
| `time_s` | toplam süre | |

Mod kararı sınıf başına verilir (sınıf dosyası `design_mode`); belirleyiciler `thin_line_recall`, `cc_added_*`, `cc_missing_pct`, `line_jaggedness_delta`, `banding`, `noise_fidelity`, `qc_score`.

### QC kalibrasyonu (`eval/calibrate_qc.py`)
Girdi `eval/score_sheet.csv` (`job_id, design_class, mode, regularize, designer_score, designer_score_2, fix_minutes, fix_type, notes`) + `qc.json`. Çıktı `models/qc_calib.json`:
```json
{"features": ["rerender_edge_iou", "..."], "dropped_features": ["banding"],
 "class_levels": ["classic", "geometric", "gradient", "other"],
 "calibrated_configs": [["classic", "segment", true], ["geometric", "segment", true],
                        ["gradient", "segment", true], ["distressed", "quantize", false], ["minimal", "segment", true]],
 "mean": [...], "std": [...], "coef": [...], "cutpoints": [...],
 "target": 4, "qc_accept": 0.82, "qc_reject": 0.31,
 "cv_auc": 0.84, "cv_auc_by_class": {"classic": 0.86, "geometric": 0.85, "gradient": 0.71, "other": 0.78},
 "class_unreliable": [], "intra_rater_kappa": 0.72,
 "n_samples": 110, "class_hist": {"classic": 40, "geometric": 30, "gradient": 18, "distressed": 12, "minimal": 10},
 "score_hist": {"1": 12, "2": 30, "3": 28, "4": 22, "5": 18},
 "fix_type_hist": {"geometry": 20, "color": 35, "regularity": 18, "motif": 12, "gradient": 9, "noise": 6, "sheen": 10},
 "model_hashes": {"segment": "...", "quantize": "..."}, "date": "..."}
```
`fix_type_hist`'te `sheen` payı yüksekse parlama/hayalet motif sorunu (otomatik çözümü olmayan) yatırım listesine girer.

---

## Süre ölçümü (bench/)

`bench/infer.py`: aşama başına süre, mod başına, 24 GB kart, üç halı boyutu → `bench/timings.json`. Süre bütçesi ve mod düşürme mekanizması yoktur: `segment` ve `quantize` 200×300 cm'de 1 dakikanın altındadır; `timings.json` sadece rapor ve regresyon takibi içindir (`time_s` eval metriği).

---

## Testler (tests/, pytest)

- `test_color.py`: sRGB→Lab, ΔE2000 referans çiftleri.
- `test_intake.py`: üç `density_definition` formülü; `grid_input_mismatch`; `reed_per_m: null` → kare varsayımı + `grid_unverified`; `grid_rounding`; `machine_colors ∉ allowed` hata; gerçek işte `design_class` eksik → hata, `--class auto` sadece batch/synthetic; `photo_below_min_px`; parametre katmanı; `setup` bloğu.
- `test_rectify.py`: sentetik köşe hatası < 8 px (`@pytest.mark.gpu`); `corners_override`; TPS bozulmuş sentetikte `non_planar`; saçak ± 3 px; çıktı tam `4W × 4H`; aydınlatma düzleştirme; `rectified.png` CLAHE içermez; `edge_width_knots` ± 0.3; `pile_blur`; efektif çözünürlük kapıları.
- `test_structure.py`: `layout_gt` ile bordür/periyot/tekrar; köşe blokları `regions == 4`; panel desende bordür kabul edilmez; `coarse_labels.npy` W×H boyutunda ve `regions.npy` bileşen sınırları buradan (512 px maske ile fark ≤ 1 düğüm); gradyan maskesi IoU ≥ 0.8; sınıf tespiti `pure` prosedürel setlerde ≥ %90; gradyanlı+gürültülü sentetikte `distressed` seçilir ve `class_mixed` tetiklenir; `pile_blur` varken `distressed` seçilmez; kamera gürültülü temiz desende `noise_frac` eşik altı kalır, `noise_1knot_frac` yükselir; `class_conflict` desenci sınıfı farklıyken tetiklenir; `knot_grid.visible`, `grid_mismatch`, `measured_reed_per_m` ± %3; TPS; `unmeasurable`.
- `test_palette.py`: komşu benzer kümeler ayrı, komşu olmayan benzerler birleşir; motif rengi korunur; gradyan uçları palette; `palette_used = 2` iki renkli desende; `palette_used ≤ machine_colors`; `gain_stop_select` ve `merge_cost_stop` ayrı yollarda ayrı ayrı durdurur; `regions.npy` girdi değildir.
- `test_design.py`: `segment` çıkış girdi/4; karo birleştirme ΔLab ortalama < 0.5, p95 < 2.0; determinizm toleranslı; sentetik gradyan bölgede boundary < 0.3; model yokken `segment_fallback` → quantize.
- `test_production.py`: 3 renk kuantizasyon; pooling eşitlik kuralı; 1×1 ve 1×2 ada silinir, 1×10 şerit korunur; `is_thin`; iki geçiş üçüncü giriş oluşturmaz; kapsama birleştirmesi motif rengini silmez; gradyan maskesinde temizleme atlanır; `min_region_knots: 0` atlar; `min_region_knots: 1` sadece tek düğümlük adaları siler; `noise_1knot_pct` doğru; `stages/guard.json` yazılır.
- `test_guard.py`: alanı > `motif_min_knots` bileşen kaybolunca sadece o bileşen geri yazılır; ≤ eşik geri yazılmaz; 11–20 düğüm bandı geri yazılır; gradyan maskesi ve distressed'ta no-op; `guard_heavy`; aşama kırılımı.
- `test_regularize.py`: tırtıklı 1:1 diagonal → temiz basamak; ±1 gürültülü düz çizgi → düz; yay artık ≤ tolerans; serbest parça run-length monoton parçalı sabit, uçlar sabit; tekrar periyotlarında tek düğüm hatası silinir; köşe bloğu dokunulmaz; `repeat_mismatch = 0` ve şablon bir kez düzleştirilir; kapsanmayan düğüm `indexed_raw` indeksini korur; `is_thin` iskelet korunur; eşitlikte en düşük indeks; `regularize: off` istatistik üretir; truth desenlerde `changed_pct ≈ 0`; `segment_classes` toplamı 1.
- `test_dither.py`: sentetik rampa → 4×4 blok karışım oranı ± 1/levels; deterministik; maske dışına taşmaz; sınır şeridi uç indekse; `dither_pattern: null` → uyarı; `dither_stats.py` düzeni doğru çıkarır.
- `test_qc.py`: truth'tan üretilen desende `rerender_edge_iou` yüksek; `qc_calib.json` yokken `needs_review`; sert kapılar (`class_mixed`, `mode_uncalibrated` dahil); null kuralları; `status` `@target`; `reasons` 3 katkı; hash uyuşmazlığı; `class_unreliable`.
- `test_calibrate.py`: ordinal model; `qc_target` seçimi; `class_levels`; > %50 null özellik düşer; kappa; sınıf başına AUC; `calibrated_configs` yazılır; kapasite medyan/p90.
- `test_align.py`: bilinen homografi ECC ile geri bulunur; yerel bozulmada `align_mask`; truth dither → `gradient_mask`; truth gürültü → `noise_mask`.
- `test_prepare.py`: `distressed` çift eğitime girer; `noise_mask` kapsaması > 0.2 karolar atlanır, kalanlar `loss_mask` ile; `loss_mask = align ∧ ¬noise`.
- `test_metrics.py`: `cc_added_*`, `cc_missing_pct`; kuantizasyon artığı (3–10 düğüm) artırmaz; `thin_line_recall`; `line_jaggedness`; `repeat_mismatch`; `banding` bantlı > dither'lı; `noise_fidelity`: gürültü silinmiş çıktıda düşük, birebir çıktıda 1, fotoğraf gürültüsü eklenmiş çıktıda `noise_cc_ratio > 1`; `edge_fidelity` tek fonksiyon.
- `test_tools.py`: `motif_stats`, `dither_stats`, `noise_stats` (sentetik eskitme deseninde `noise_cc_min/max` ve tek düğüm payını doğru çıkarır), `pair_stats` (`dist.yaml`), `pair_intake` (eksik alan/düşük çözünürlük reddi), `setup-check` çıktısı.
- `test_generate.py`: prosedürel `pure` ve `mixed` örnekler meta ile tutarlı; `dist.yaml` örnekleme aralıkları.
- `test_export.py`: `design.tiff` yazılıp okunur; anizotropik `resolution`; indeks sayısı ≤ `machine_colors`; `palette.csv` sütunları.
- `test_cache.py`: aynı girdi ikinci `run` hiçbir aşamayı çalıştırmaz; `rerun --from production --palette` sadece production ve sonrası; sınıf dosyası değişince `structure` sonrası yeniden çalışır.

---

## Kod yapısı

```
knotvision/
├── config/
│   ├── looms/wilton_default.yaml
│   ├── classes/{classic,geometric,gradient,distressed,minimal}.yaml
│   ├── dither/{ordered4.yaml, archive.yaml}
│   └── yarns/sample.csv (optional)
├── knotvision/
│   ├── job.py          Job dataclass, runs/, hash cache, retention, rerun, parameter layering, setup block
│   ├── color.py        sRGB/Lab, deltaE2000
│   ├── intake.py       density definitions -> grid, validation, photo resolution check
│   ├── rectify.py      SAM 3.1 wrapper, manual corners, fringe, planarity, homography, white balance, illumination, edge sharpness
│   ├── structure.py    layout, border acceptance, gradient mask, coarse labels at grid resolution, class detection, knot grid field, TPS, regions
│   ├── palette.py      photo palette (adjacency-aware merge, gradient endpoints, motif-color protection), yarn library matching
│   ├── design_seg.py   SegNet (convnext_tiny + FPN, stride 4), grid-aligned tiling
│   ├── design_quant.py 4 px/knot mean-shift
│   ├── production.py   two-pass quantization, mode pooling, noise measurement, cleanup, is_thin, coverage merge
│   ├── guard.py        assert_no_motif_loss, stages/guard.json
│   ├── regularize.py   template-first repeat/symmetry merge, vectorization, angle snap, arc fit, free-run regularization, re-rasterization
│   ├── dither.py       analytic ordered dither inside the gradient mask
│   ├── metrics.py      edge_fidelity, line_jaggedness, repeat_mismatch, banding, noise_fidelity, cc_*, thin_line_recall
│   ├── qc.py           hard gates, rerender, features with null rules, ordinal calibrated score, reasons
│   ├── export.py       design.tiff + palette.csv
│   └── run.py
├── tools/ motif_stats.py, dither_stats.py, noise_stats.py, pair_stats.py, pair_intake.py
├── data/ (real, archive, backgrounds, synth)
├── train/ prepare.py, align.py, train_segment.py, align_report.csv, runs/
├── eval/ run.py, calibrate_qc.py, capacity.py, score_sheet.csv, synth_test_ids.txt, synth_modern_ids.txt, results/
├── bench/ infer.py, timings.json
├── tests/
├── models/ knotvision_segment.pt, qc_calib.json (gitignore)
├── cli.py   run | rerun --from <stage> [--class c] [--palette ...] [--confirm-palette] [--mode ...] [--corners ...]
│            [--no-gate] [--regularize on|off] [--dither on|off] | setup-check | batch --class auto
├── requirements.txt
└── CLAUDE.md
```

## Bağımlılıklar

```
numpy, opencv-contrib-python, pillow, scikit-image, scipy, scikit-learn, statsmodels, tifffile, pyyaml
torch, torchvision, timm
segmentation-models-pytorch   # optional, fallback decoder only
sam3   # pip install git+https://github.com/facebookresearch/sam3
noise
pytest
```

## Uygulama sırası

Kod yazmayı bekleten dış girdi yoktur; kurulum verisi geldikçe varsayılanların yerine geçer.

1. **Kurulum ve arşiv araçları:** tezgah profili; varsa iplik CSV. `motif_stats.py`, `dither_stats.py`, `noise_stats.py` arşiv üzerinde → `motif_min_knots`, `config/dither/archive.yaml`, `noise_cc_min/max`, distressed `min_region_knots`; desenciye teyit. `setup-check` ile eksikler listelenir.
2. `generate_designs.py` (prosedürel, karışık özellikli) — arşive bağımlı değil, ilk gün; `test_core_modern` sentetik sürümü buradan. `dist.yaml` varsayılanlarla.
3. Modelsiz uçtan uca hat: `intake → rectify → structure → palette → quantize → production → regularize → dither → qc (özellikler, karar yok) → export`. Texcelle'de açılır. `reed_per_m: null` ise ilk işlerin `measured_reed_per_m` değeriyle profil doldurulur.
4. **Çift toplama başlar** (`pair_intake.py` ile, sınıf kotalı). İlk 20–30 çiftte `align.py` → `align_report.csv`; elenme oranı yüksekse kriter düzeltilir. `pair_stats.py` → `dist.yaml`.
5. **Desenci ölçümü (model eğitiminden önce):** 10 gerçek iş — 3 `classic`, 3 `geometric`, 2 `gradient`, 1 `distressed`, 1 `minimal` — modelsiz hattan, `regularize` açık/kapalı iki varyant. Süre, puan, düzeltme türü (`sheen` ve `noise` dahil). Bu ölçüm: projenin işe yarayıp yaramadığını, yatırımın nereye gideceğini, kapasite formülünün ilk girdisini, `jag_max`, `banding_max`, `deltaE_warn` başlangıçlarını verir. Sonuç gelmeden 8. adıma geçilmez.
6. `bench/infer.py` → `timings.json`.
7. `generate.py` + offline eval, sınıf kırılımlı. Modelsiz hat = taban çizgisi.
8. `segment` eğitimi (prosedürel set + hizalanan gerçek çiftler, karo bazlı atlama). Eval; yayın kriterine iki çekirdek sette bak. `distressed`'ta `noise_fidelity` quantize'ı geçerse sınıf dosyası güncellenir. `free_len_share` klasikte > 0.6 ise v7 öğrenilmiş vektörleştirme gündeme alınır.
9. QC kalibrasyonu: sınıf kotalı ≥ 100 çıktı, intra-rater kontrolü → `qc_calib.json`. Bundan sonra pipeline karar verir.
10. SAM 3.1 entegrasyonu (erişim geldiğinde).

## Sadeleştirme kararları

- **`edit` modu spesifikasyondan çıkarıldı.** Gerekçe: (a) henüz taban çizgisi ve desenci ölçümü yokken 20B diffusion modeli, LoRA eğitimi, döşeme kapıları ve süre bütçesi planlayıcısı spesifikasyonun yaklaşık dörtte birini işgal ediyordu; (b) `edit` hiçbir sınıfın varsayılanı olmadığından her çıktısı sert kapıdan `needs_review`'a düşüyordu ve kendi kalibrasyonu tanımsızdı; (c) 4 px/düğüm sabit çözünürlük kuralının tek istisnasıydı. v7 adayı olarak duruyor; `segment` yayın kriterini tutturamazsa geri alınır ve `calibrated_configs` mekanizmasıyla kalibre edilir.
- **Süre bütçesi ve mod düşürme yok.** Kalan iki mod 1 dakikanın altındadır. `bench` sadece rapor.
- **Tek sınıf etiketi.** Karışık halıyı sınıf değil, sınıftan bağımsız gradyan maskesi ve `class_mixed` uyarısı taşır.
- **Sınıf desenciden.** Gerçek işte otomatik tespit karar vermez, rapor eder ve çelişkiyi işaretler.
- **Gradyan maskesi muafiyetleri sınıf dosyasında değil, aşama kodunda.** Her sınıfta aynı; sınıf dosyası sadece maske dışı davranışı ve QC setini değiştirir.
- **İşlem iplikleri pipeline dışı.** Palet makinenin renk sınırının tamamını kullanır.
- **`regions.npy` tek tüketicili (`regularize`).** `palette` girdisinden çıkarıldı; orada kullanılmıyordu.
- **`gain_stop` ikiye ayrıldı.** Kütüphaneli yolda kazanç, kütüphanesiz yolda maliyet; birim ve yön farklı.
- **Açık nokta yerine kurulum.** Varsayılanla çalışan her alan sert kapı üretir, sessizce geçmez; `setup-check` listeler.
- **Parlama / hayalet motif için mekanizma yok.** Fotoğraftan çözülemeyen sorun için ölçüm (`fix_type: sheen`) var, kod yok.

## v7 adayları

`edit` modu (diffusion + LoRA, kendi kalibrasyon konfigürasyonuyla), retrieval (DINOv2), rulo tekrar, `pictorial` sınıfı, diğer tezgah profilleri, sınır düğümleri için sınıflandırma başlığı, palet koşullu SegNet başlığı, QC için öğrenilmiş geri-render, öğrenilmiş vektörleştirme (`free_len_share` > 0.6 ise), gradyan bölgesinde öğrenilmiş rampa kestirimi, parlama/hayalet motif için desenci girdili motif işaretleme (`sheen` payı yüksekse).
