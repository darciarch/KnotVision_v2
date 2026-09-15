# KnotVision Pipeline v6.6

Müşteri fotoğrafı → Texcelle'de açılan, makinenin renk sınırına uyan, üretime giden indeksli desen dosyası.

Kapsam: Wilton, parça halı. Desen üretimi üç mod:
- `quantize` — modelsiz, deterministik taban çizgisi. **Profil varsayılanı.** `distressed` sınıfının kalıcı varsayılanı. Her modda ilk çalışan yoldur: `edit` için referans ve yedek, `segment` için hata yedeği.
- `segment` — deterministik ağ, ızgara çözünürlüğünde. Bir sınıfın varsayılanı olması için yayın kriterini o sınıfta tutturması gerekir; tutturunca sınıf dosyasına yazılır.
- `edit` — diffusion tabanlı görüntü düzenleme modeli, **agent** ile çalıştırılır (aday üretimi → ölçüm → geri bildirimli yeniden düzenleme → seçim). Birden fazla agent tanımlanır ve yarıştırılır (`eval/agents.py`); yarışı kazanan agent, sınıfın yayın kriterini tutturursa o sınıfın varsayılanı olur. Arka uçlar: `qwen_edit_2511` (açık ağırlık, yerel GPU), `qwen_edit_2509` (ablasyon, opsiyonel), `muse_image` (Meta Model API, ücretli). Hiçbir sınıfın başlangıç varsayılanı değildir; sadece `calibrated_configs`'a giren `(sınıf, edit, agent)` üçlüleri karar alır.

Kapsam dışı:
- Retrieval, rulo tekrar, `pictorial` sınıfı (manzara, portre, foto-halı), Wilton dışı tezgahlar.
- **İşlem iplikleri** (oyma, kesme, parlaklık). İşlem bir iplik renginin tamamına uygulanır; Texcelle'de desenci tarafından ipliğe etiket olarak girilir. Desen dosyasında ayrı indeks gerektirmez, fotoğraftan tespit edilmez. Pipeline'ın işi deseni doğru renkle çizmektir.

Pipeline'ın davranışı **desen sınıfı** (`design_class`) ile belirlenir; sınıfın kaynağı gerçek işlerde desencidir. Sınıf; temizleme eşiklerini, düzenlileştirmenin açık/kapalı olmasını, zorlama bayraklarını ve QC özellik setini seçer. Dither sınıftan değil, gradyan maskesinin varlığından türer. Mod (`design_mode`) sınıf dosyasından gelir, iş override'ı ile ezilir.

Hedef: fotoğrafa olabildiğince yakın, **desenci elinden çıkmış gibi düzenli** desen. Kalite tanımı dört parçadır ve dördü de ölçülür:
1. Sadakat — motifler doğru yerde, doğru renkte, yoktan motif eklenmemiş, var olan motif silinmemiş.
2. Düzenlilik — çizgiler düz, diagonaller temiz basamaklı, tekrar eden motifler birebir aynı, simetri birebir. `regularize` üretir; `line_jaggedness`, `repeat_mismatch` ölçer.
3. Renk geçişi — gradyanlı bölgeler bantlaşmadan, desencinin dither düzeniyle. `dither` üretir; `banding`, `dither_err` ölçer.
4. Gürültü sadakati — eskitme desenlerinde kasıtlı gürültü korunur, fotoğraf kaynaklı gürültü (hav dokusu, iplik sapması, kamera/JPEG) düğümlere yazılmaz. `noise_fidelity` ölçer (eval), `noise_1knot_pct` izler (QC).

Pipeline çıktısı desencinin düzelttiği bir taslak değil, üretime giden nihai dosyadır. Yoktan motif eklenmesi ve var olan motifin silinmesi kabul edilmez. Üç ayrı mekanizma vardır:
- Offline: `cc_added_motif = 0` (`edit`'te `edit_added_motif_max`, başlangıç 0) ve `cc_missing_pct ≤ 0.01`, referans `truth.tiff`.
- Online, indeks haritası olan aşamalarda: `guard` bileşen geri yazar.
- Online, `edit` modunda `design` çıktısı için: `quantize` taban çizgisine karşı `edit_added_pct` / `edit_missing_pct`, ikisi de sert kapı (bkz. `design` / edit).

İki ayrı kontrol düzeyi:
1. Offline değerlendirme — model iyi mi (seyrek, referanslı).
2. Online kalite kontrolü — bu iş desenciye gösterilmeye değer mi (her iş, referanssız).

Lisans kuralı: yerel çalışan her model ve kütüphane ticari kullanımda ücretsiz olmalı. Kabul: Apache 2.0, MIT, BSD, SAM License. Ret: AGPL (ultralytics), "research only" ağırlıklar. Ücretli API arka uçları (`muse_image`) kabul edilir; şartları: (a) profilde `allow_api: true`, (b) iş başına maliyet `cost_usd` raporlanır, (c) müşteri fotoğrafının dışarı çıktığı `api_egress: true` ile raporlanır, (d) API tarafında eğitim/LoRA yoktur.

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
| Parlama / hayalet motif | Renk farkıyla değil hav yatışı veya işlemle görünen motif fotoğraftan ayırt edilemez: ayrı iplik ama RGB'si yakın (palet birleştirir, motif kaybolur), tek renge işlem (palete boşa indeks açılır) veya sadece hav artefaktı (yoktan motif çizilir). Aydınlatma düzleştirmesi bunu çözmez; parlama motif şeklini izler, düşük frekanslı değildir. | Otomatik çözüm yoktur. `score_sheet.csv`'de `fix_type: sheen`; desenci ölçümünde sıklığı ölçülür. Sıklık yüksekse sonraki sürümde ele alınır. |
| Fotoğraf çözünürlüğü | Eskitme ve detaylı desenler 1–2 düğüm ölçeğinde bilgi taşır. 160×230 cm, 400 çözgü/m → 640×920 düğüm; 4 px/düğüm için 3680 px uzun kenar gerekir. Müşteri fotoğrafları çoğunlukla telefon/WhatsApp üzerinden 1600–2000 px uzun kenarla gelir; bu 1.7–2.2 px/düğüm demektir ve bulanık kenar efektif değeri daha da düşürür. | Tek kapı **efektif çözünürlük**tür (`photo_px_per_knot_eff`; `photo_px_warn` 1.5, `photo_px_stop` 0.8). Uzun kenar piksel sayısı için sert eşik yoktur; `photo_long_px_rec` (3000) sadece müşteriye önerilen değerdir. Çift toplamada aynı kural: `pair_intake.py` geometrik px/düğüm < `photo_px_stop` ise reddeder, altındakileri uyarır. Eldeki müşteri fotoğraflarının dağılımı kod yazılmadan önce ölçülür (`tools/photo_survey.py`); sonuç yeniden fotoğraflama kararını belirler. |
| Sapma | Desencinin nihai deseni fotoğraftan sapabilir; saha bilgisi "çok küçük". | `align_report.csv` ile ölçülür; `cc_added_motif = 0` sert kural. |
| Desenci pratiği | Düz çizgi çizer, aynı motifi birebir kopyalar, diagonalleri temiz basamaklandırır. Bordür köşesi kenar şeridinin periyoduna uymayan ayrı motiftir. | `regularize`; köşe blokları muaf; `line_jaggedness`, `repeat_mismatch`. |
| İplik RGB | Desenci paleti kendi seçer, Texcelle'de gerçek ipliklerle eşler. Kütüphane RGB'leri yaklaşıktır. | İplik kütüphanesi opsiyonel. Palet her durumda öneridir (`palette_confirmed`). |
| Hav yönü / ışık | En büyük renk hatası havın yatış yönüdür. | `rectify`'da düşük frekanslı çarpımsal aydınlatma düzleştirme; sentetikte hav yönü bağımlı gölgelenme. |
| Uzun hav / odak | Motif kenarı 3–4 düğüme yayılır; piksel/düğüm yüksek olsa bile efektif çözünürlük düşüktür. | `edge_width_knots`, `photo_px_per_knot_eff`, `pile_blur`. |
| Regularize kazancı | Klasik motif (yaprak, rumi, palmet) çoğunlukla serbest parçaya düşer; açı kilidi ve çember fit asıl kazancı geometrikte, bordür çerçevesinde ve diagonalde verir. | `free_len_share` sınıf kırılımıyla raporlanır; klasikte > 0.6 ise öğrenilmiş vektörleştirme gündeme alınır. |
| İnce çizgi | 1 düğüm genişliğinde çizgiler (`line_max_width: 1`) klasik ve geometrik desenlerde yaygındır. Diffusion düzenleme modeli ≤ 2 px/düğüm'de çalışır ve VAE 8× altörnekler; 1 düğümlük çizgi bu çözünürlükte ne görülür ne çizilir. | `structure` `thin_line_frac` ölçer; `edit` bu değerin üstündeki işlere **yönlendirilmez** (`edit_thin_line_max`, `edit_thin_lines` uyarısıyla sınıfın edit dışı varsayılanına döner). `thin_line_recall` kriteri mod başına, o modun gerçekten çalıştığı iş kümesinde ölçülür. Karolama (`edit_tiling: auto`) tek geçişte 2 px/düğüm'e ulaşılamayan halılarda varsayılan olarak açılır. |
| Diffusion düzenleme riski | Düzenleme modeli üç hata üretir: (a) geometri kayması — motifler birkaç piksel/düğüm kayar, ölçek değişir; (b) uydurma — fotoğrafta olmayan motif/detay ekler, silik motifi "tamamlar", var olanı siler; (c) çözünürlük — model 1–2 MP'de çalışır, 15 MP'lik çalışma çözünürlüğünü tek geçişte üretemez. Çıktı seed'e bağlıdır; API modelinde seed yoktur. | (a) `edit` sonrası ECC hizalama (`edit_align_px`), `edit_drift` sert kapı; (b) referans = aynı işin `quantize` taban çizgisi (`ref_indexed.npy`, aday ile aynı palet ve kuantizasyon yolundan geçer; hav/gölge gürültüsü büyük ölçüde elenir), `edit_added_pct` / `edit_missing_pct` ikisi de sert kapı, eval'de `cc_added_*` / `cc_missing_pct` ile korelasyon ölçülür ve eşikler oradan sabitlenir; (c) `edit` kendi çözünürlüğünde (≤ 2 px/düğüm) çalışır, karolama otomatik, çıktı 4 px/düğüm'e büyütülür, `production` kuantize eder. Seed profilde sabit, cache seed'i içerir; `deterministic: false` raporlanır. Agent döngüsü bu hataları ölçüp geri bildirim olarak prompt'a yazar. |

---

## Kurulum

Fabrikadan fabrikaya ve desenciden desenciye değişen her şey config'dir. Yeni kurulumda üç dosya, bir arşiv taraması ve bir fotoğraf taraması:

| Ne | Dosya | Kaynak | Yoksa (pipeline yine çalışır) |
|---|---|---|---|
| Tezgah | `config/looms/<fabrika>.yaml` — `reed_per_m`, `density_definition`, `allowed_machine_colors`, `allow_api` | fabrika | `reed_per_m: null` → kare varsayımı, `grid_unverified` sert kapı; ilk işteki `knot_grid` ölçümü profili doldurur |
| Fotoğraf taraması | `data/photo_survey.csv` — eldeki müşteri fotoğraflarının uzun kenar px, ebat, geometrik px/düğüm dağılımı | `tools/photo_survey.py` | çift toplama hedefi ve yeniden fotoğraflama kararı verilemez; `setup-check` uyarır |
| Dither düzeni | `config/dither/<ad>.yaml` | `tools/dither_stats.py` arşivden, desenci teyit eder | 4×4 ordered, `dither_pattern_default` sert kapı (sadece gradyan maskesi olan işlerde) |
| İplik kütüphanesi | `config/yarns/<desenci>.csv` | desenci | palet fotoğraftan, `yarn_code` boş |
| Motif eşiği | profilde `motif_min_knots` | `tools/motif_stats.py` arşivden | 10, `motif_min_knots_source: assumed` |
| Gürültü ölçeği | profilde `noise_cc_min`, `noise_cc_max`, distressed `min_region_knots` | `tools/noise_stats.py` arşivdeki eskitme desenlerinden | 2 / 6 / 1, `noise_scale_source: assumed` |
| QC kalibrasyonu | `models/qc_calib.json` | `eval/calibrate_qc.py` | skor hesaplanır, karar verilmez (`needs_review`) |
| Segment ağırlığı | `models/knotvision_segment.pt` | `train/train_segment.py` | sınıf dosyasında `design_mode: segment` yazılı olsa da `quantize` çalışır, `segment_fallback`; sınıf dosyalarına `segment` ancak eğitim ve eval sonrası yazılır |
| Edit agent'ları | `config/agents/<ad>.yaml` | repo (`qwen2511_direct`, `qwen2511_loop`, `muse_loop` hazır gelir) | `design_mode: edit` istenirse `edit_agent` zorunlu; yoksa hata |
| Edit arka uç ağırlıkları | `models/qwen_image_edit_2511/` (HF'den indirilir), opsiyonel `models/edit_lora/<ad>.safetensors` | HF, `train/train_edit_lora.py` | `qwen_edit_*` arka ucu `edit_backend_error` → taban çizgisi (`quantize`) kullanılır, `warnings += "edit_fallback"` |
| Muse API anahtarı | ortam değişkeni `MUSE_API_KEY` (Meta Model API veya sağlayıcı: fal / Runware / OpenRouter; `muse_provider` profilde) | fabrika hesabı | `muse_image` arka ucu `edit_backend_error` |

Kurulum eksikliği sistematik `needs_review` üretir: `reed_per_m: null` her işte, `dither_pattern: null` gradyan maskesi olan her işte, `qc_calib.json` yokluğu her işte. Bunlar dolmadan otomatik kabul yoktur; `cli.py setup-check` eksik alanları ve her birinin hangi kapıyı tetiklediğini listeler; `edit` arka uçlarının erişilebilirliğini (ağırlık dosyası, API anahtarı, sağlayıcıya test isteği) de kontrol eder.

Kurulum durumu her işin raporunda `setup` bloğuna yazılır: `{reed_per_m: profile | assumed, density_definition, dither_pattern: designer | archive | default, yarn_library: bool, motif_min_knots: archive | assumed, noise_scale: archive | assumed, qc_calib: bool, sam: bool, segment_ckpt: bool, edit_backends: {qwen_edit_2511: bool, qwen_edit_2509: bool, muse_image: bool}}`.

**SAM 3.1 erişimi alındı** (HF gated onayı). Checkpoint `models/sam3.1/`; kurulum bağımlılığıdır, `setup-check` doğrular. Çalışma zamanı hatasında (checkpoint bozuk, VRAM yetersiz) `sam_unavailable` → `needs_corners`; `corners_override` ile manuel köşe her zaman geçerlidir.

---

## Desen sınıfı (`design_class`)

`design_class ∈ {classic, geometric, gradient, distressed, minimal}`. `pictorial` tespit edilirse iş `class_unsupported` ile durur (desenci `--class <c>` ile ezerse `needs_review` ile devam eder).

Kaynak:
- **Gerçek iş:** desenci girdisi zorunlu (`class_source: designer`). Otomatik tespit yine çalışır; sadece `class_auto`, `class_margin` ve `class_features` raporlanır. Otomatik sınıf desenciyle çelişiyor ve `class_margin ≥ class_margin_min` ise `warnings += "class_conflict"` (yumuşak; karar desencinindir).
- **Sentetik veri ve toplu değerlendirme:** `--class auto` (`class_source: auto`). `class_margin < class_margin_min` → `class_uncertain`, sert kapı `needs_review`.

Sınıflar gerçek halılarda birbirini dışlamaz (degrade zeminli madalyon, gradyanlı kafes, hem eskitmeli hem gradyanlı modern). Çoklu etiket **kullanılmaz**: parametre katmanı tek sınıf dosyası okur. Bunun yerine sınıftan bağımsız çalışan mekanizmalar karışık halıyı taşır: gradyan maskesi (dither, maske içi muafiyetler, palette uç renkleri) sınıfa bakmaz. Sınıfa bağlı kalan tek şey `design_mode` (ve `edit` ise `edit_agent`) ve maske dışındaki `min_region_knots` / `min_coverage_pct` / `regularize` / `force_*`. `gradient_frac` ve `noise_frac` ikisi birden eşik üstündeyse `warnings += "class_mixed"`, sert kapı `needs_review` (hangi kaynaktan gelirse gelsin).

### Parametre katmanı

Her parametre üç katmandan çözülür, sonraki öncekini ezer: tezgah profili ← sınıf dosyası (`config/classes/<design_class>.yaml`) ← iş override'ı. `design_mode: edit` çözüldüğünde dördüncü katman devreye girer: agent dosyası (`config/agents/<edit_agent>.yaml`) sadece `edit_*` alanlarını ezer, sınıf dosyası ile iş override'ı arasına oturur (profil ← sınıf ← agent ← iş). Çözülmüş set `job.json.resolved_params`; hash'i aşama cache anahtarına girer. Sınıf dosyaları `allowed_machine_colors`, `reed_per_m`, `density_definition`, `retention_days`, `allow_api` dışında her alanı ezebilir.

Mod çözümünden sonra iki yönlendirme kuralı uygulanır (`structure` sonunda, çünkü girdileri orada hesaplanır):
- `design_mode: segment` ve `models/knotvision_segment.pt` yok → `quantize`, `warnings += "segment_fallback"`.
- `design_mode: edit` ve `thin_line_frac ≥ edit_thin_line_max` → sınıfın edit dışı varsayılanı (`segment` varsa o, yoksa `quantize`), `warnings += "edit_thin_lines"`. Yönlendirme sert kapı değildir; sonuç modun kendi kalibrasyon durumu geçerlidir.

### Sınıfların değiştirdiği parametreler

| Sınıf | Sınıf katmanı olmasa kırıldığı yer | Sınıf dosyasının ezdiği alanlar |
|---|---|---|
| `classic` | — | varsayılanlar (eğitim sonrası `design_mode: segment`) |
| `geometric` | — (regularize burada en iyi çalışır) | varsayılanlar; `force_symmetry: auto` kalır (eğitim sonrası `design_mode: segment`) |
| `gradient` | maske dışı davranış varsayılanla aynı; sınıf sadece QC özelliğini ekler ve raporlar | `qc_features_extra: [banding]` (eğitim sonrası `design_mode: segment`) |
| `distressed` | temizleme, kapsama birleştirme ve regularize kasıtlı gürültüyü siler; `noise_knots_pct` yüksek çıkar ve doğru çıktı reddedilir; segment boundary kanalı doyar | `design_mode: quantize`, `regularize: off`, `min_region_knots: 1` (sadece tek düğümlük adalar), `min_coverage_pct: 0`, `force_*: off`; `noise_knots_pct`, `cleanup_changed_pct` null |
| `minimal` | kenar yokluğunda `rerender_edge_iou` ve `line_jaggedness` gürültü | `edge_density < edge_density_min` ise bu iki özellik null; `force_symmetry: off` |

Sınıf dosyasına `design_mode: segment` veya `design_mode: edit` + `edit_agent: <ad>` yazılması elle değil ölçümle olur: `segment` için o sınıfta yayın kriteri, `edit` için agent yarışı sonucu (commit mesajında eval CSV referansı). Panel / çerçeveli desen ayrı sınıf değildir; bordür kabul koşuluyla ele alınır (`structure` adım 2).

### Otomatik tespit (`structure` adım 6)

Girdi büyüklükleri `structure`'ın zaten hesapladıklarıdır:
- `edge_density`: `rectified_contrast` (512 px) Canny kenar piksel oranı.
- `n_colors_est`: 400 px Lab k-means (k = 16, sabit seed), merkezler ΔE < 8 ise birleştirilir, kalan küme sayısı.
- `gradient_frac`: gradyan maskesinin alan oranı (adım 5b).
- `noise_frac`: `coarse_labels.npy` (adım 6, ızgara çözünürlüğü) üzerinde alanı ∈ `[noise_cc_min, noise_cc_max]` düğüm olan bileşenlerin düğüm oranı. Tek düğümlük bileşenler **sayılmaz**; onlar `noise_1knot_frac`'a gider. Gerekçe: kasıtlı eskitme çok düğümlük lekedir, tek düğüm fotoğraf gürültüsüdür.
- `thin_line_frac`: `coarse_labels.npy` üzerinde `is_thin` (bkz. `production` adım 5) bileşenlerin düğüm oranı.
- `symmetry_conf`, `border_period_conf`, `ground_repeat.conf`, `medallion`.
- `line_frac`: kenar piksellerinin `HoughLinesP` doğrularıyla kapsanan oranı.

Kurallar sırayla, ilk eşleşen:
1. `gradient_frac ≥ 0.6` ve tekrar/simetri yok ve `edge_density ≥ edge_density_min` → `pictorial`. (Kenarsız düz degrade halı bu kurala girmez, kural 3'e düşer ve `gradient` olur.)
2. `noise_frac ≥ noise_class_frac` (0.08) ve `pile_blur` yok → `distressed`.
3. `gradient_frac ≥ gradient_class_frac` (0.25) → `gradient`.
4. `n_colors_est ≤ 3` ve `edge_density < edge_density_min` → `minimal`.
5. `symmetry == hv` veya (bordür kabul + `medallion`) → `classic`.
6. `ground_repeat` tespit veya `line_frac ≥ 0.6` → `geometric`.
7. aksi → `classic`.

**Marj (`class_margin`)**: her kuralın her koşulu bir orana çevrilir: `≥ eşik` koşulu için `değer / eşik`, `< eşik` koşulu için `eşik / değer`, boolean koşul için ilgili güven / tespit eşiği (0.6). Bir kuralın ateşleme oranı `fire_i` = koşullarının en küçüğü; kural `fire_i ≥ 1` ise eşleşir. Eşleşen kural 1–6 ise `class_margin = fire_i − 1`; kural 7 ise `class_margin = 1 − max(fire_1 … fire_6)` (en yakın kuralın ateşlemeye ne kadar uzak kaldığı). `class_margin < class_margin_min` (0.2) → `class_uncertain`. Marj eşiğe normalize olduğu için her kuralda aynı ölçektedir.

Sıra gerekçesi: hem gradyanlı hem gürültülü halıda yanlış `gradient` kararı gürültüyü siler (geri alınamaz); yanlış `distressed` kararında gradyan maskesi ve dither yine çalışır, sadece mod `quantize` olur. Ucuz hata yönü öne alınmıştır. `pile_blur` koşulu: bulanık fotoğrafta yüksek `noise_frac` desen gürültüsü değildir.

Rapor: `design_class`, `class_source`, `class_auto`, `class_margin`, `class_features`, `class_conflict`, `class_mixed`.

---

## Motif koruma (`guard.py`)

İndeks haritasını değiştiren her aşamanın sonunda: `assert_no_motif_loss(before, after, params) → after'`.
- `before` ve `after` bileşenleri eşleştirilir (aynı indeks, IoU ≥ 0.3).
- `before`'da alanı > `motif_min_knots` olan ve `after`'da eşleşmeyen her bileşen geri yazılır. Aşama geri alınmaz, sadece bileşen.
- Gradyan maskesi ve `distressed`'ta harita kontrol dışıdır.
- Çağrı noktaları: `production` temizleme sonrası, `production` kapsama birleştirmesi sonrası, `regularize` adım 2 (her yapı sonrası), `regularize` adım 3. `palette`'te indeks haritası yoktur; koruma `motif_color_min_cc` ile kalır.
- `design` aşamasında indeks haritası yoktur. `edit` modunda `design` çıktısının motif kaybı `production` adım 3'te `ref_indexed.npy`'ye karşı ölçülür ve `edit_missing` sert kapısıyla durdurulur; geri yazma yapılmaz (taban çizgisi truth değildir, iki farklı render'ı karıştırmak yerine iş desenciye gider).
- Her çağrı `stages/guard.json`'a eklenir (`stage, reverted_components, reverted_area_pct`). Dosya `production` ve `regularize`'ın çıktısıdır, `qc` girdisidir.
- Toplam `reverted_area_pct > guard_warn_pct` (0.02) → `guard_heavy`, sert kapı.
- Eval'de `cc_missing_pct` aşama kırılımıyla raporlanır; hangi aşamanın motif sildiği buradan okunur.

Regularize için ayrı "değişen düğüm yüzdesi" kapısı yoktur: yüzde kapısı düzenlileştirmeye en çok ihtiyaç duyan işi engeller; guard asıl riski (motif silinmesi) doğrudan ölçer.

`motif_min_knots`: `tools/motif_stats.py` arşiv truth desenlerinde `is_thin` olmayan bileşen alanı dağılımının 5. yüzdeliği. Arşiv yoksa 10, `assumed`.

---
## Kabul ve kontrol

### 1. Offline değerlendirme (model geliştirme)

Soru: model iyi mi. Seyrek çalışır: yeni checkpoint, veri değişimi, yeni agent tanımı. Referans `truth.tiff`.

Desenci puanı ve **düzeltme süresi** kalibrasyon verisidir. Puan: 5 = değişiklik yapmadan üretime; 4 = < 30 dk; 3 = < 2 saat; 2 = sadece şablon; 1 = sıfırdan. `score_sheet.csv`: puan, düzeltme süresi (dk), düzeltme türü (`geometry | color | regularity | motif | gradient | noise | sheen`), `design_class`, `mode`, `edit_agent`.

Yayın kriteri, **iki sabit sette ayrı ayrı** (`test_core_classic`, `test_core_modern`) + 100 sabit sentetik (arşiv) + 100 sabit sentetik (prosedürel), her çiftte. Kriter **mod başına, o modun yönlendirme kurallarından sonra gerçekten çalıştığı iş kümesinde** ölçülür; `edit` için `edit_thin_lines` ile başka moda yönlendirilen işler `edit`'in kümesine girmez, yönlendirme oranı ayrıca raporlanır (`edit_route_out_rate`).
- `cc_added_motif = 0` (sert; gradyan maskesi ve truth gürültü maskesi dışında). `edit` modunda eşik `edit_added_motif_max` (profil, başlangıç 0). Bu sayı desenci ölçümünden sonra sabitlenir: desenci "benzer" çıktıyı puan ≥ 4 ile kabul ediyorsa ve eklenen bileşenler `fix_type: motif` üretmiyorsa gevşetilir; aksi halde 0 kalır. Gevşetme sadece `edit` için geçerlidir.
- `cc_added_area_pct ≤ 0.002`.
- `cc_missing_pct ≤ 0.01`.
- `thin_line_recall ≥ 0.9`.
- `knot_deltaE_mean < eval_deltaE_max`, `knot_deltaE_p95 < 2 × eval_deltaE_max` (gradyan maskesinde 8×8 blok ortalaması). `eval_deltaE_max` truth'a göre indeks hatasıdır; `production`'daki `deltaE_warn` (düğüm başına palet uzaklığı) ile aynı şey değildir ve ayrı tutulur.
- `line_jaggedness_delta ≤ jag_max` (regularize açık işlerde; başlangıç 0.15, quantize taban çizgisi sonrası sabitlenir).
- `repeat_mismatch ≤ 0.01` tekrar bölgelerinde.
- `banding ≤ banding_max` gradyan maskesi olan işlerde (başlangıç 2.0 L).
- `noise_fidelity ≥ noise_fidelity_min` truth gürültü maskesi olan işlerde (başlangıç 0.8).
- `edit` modunda ek: `edit_align_px_p95 ≤ edit_align_max`; `edit_rounds` medyanı ≤ `edit_max_rounds − 1`; `edit_added_pct` ↔ `cc_added_area_pct` ve `edit_missing_pct` ↔ `cc_missing_pct` Spearman korelasyonu ≥ `edit_gate_corr_min` (0.6) — bu koşul sağlanmadan `edit_added_max` / `edit_missing_max` eşikleri `assumed` kalır ve konfigürasyon kalibrasyona giremez; `cost_usd` medyanı raporlanır (kriter değil).
- Online QC skorunun her iki `test_core` ortalaması ≥ `qc_accept`.
- Önceki sürüme göre hiçbir metrikte, hiçbir sınıfta > %10 kötüleşme (CI, sınıf kırılımlı).

`eval_deltaE_max` başlangıç 3.0 ve yer tutucudur. Desenci ölçümünde `fix_type: color` ile korelasyona göre sabitlenir.

**Edit kapı eşiklerinin sabitlenmesi (`eval/run.py` → profil):** korelasyon koşulu sağlanınca `edit_added_max` = `cc_added_area_pct ≤ 0.002` olan işlerin `edit_added_pct` 95. yüzdeliği; `edit_missing_max` = `cc_missing_pct ≤ 0.01` olan işlerin `edit_missing_pct` 95. yüzdeliği. Profile yazılır, `edit_gate_source: eval`. Kapılar `assumed` iken de çalışır (her edit işi zaten `mode_uncalibrated` ile `needs_review`'dur); `eval` kaynaklı eşik olmadan hiçbir edit konfigürasyonu `calibrated_configs`'a giremez.

**Agent yarışı (`eval/agents.py`):** aynı setlerde her agent × her sınıf; tablo sınıf kırılımlı, sütunlar yukarıdaki metrikler + `edit_rounds` medyan + `edit_route_out_rate` + `time_s` medyan + `cost_usd` medyan. Sınıf başına kazanan: yayın kriterini tutturan agent'lar arasında `qc_score` ortalaması en yüksek olan; eşitlikte ucuz (süre, sonra maliyet). Kazanan yoksa sınıf `segment`/`quantize` varsayılanında kalır. En fazla **iki** agent QC kalibrasyonuna girer (`calibrated_configs`); gerekçe kalibrasyon örneklem maliyeti. Yarış sonucu `eval/results/agents_<tarih>.csv`.

### 2. Online kalite kontrolü (her iş, `qc` aşaması)

Soru: bu çıktı desenciye gösterilmeye değer mi. Referans yok.

**Sert kapılar** (kalibrasyon gerektirmez):
- `palette_used ≤ machine_colors`.
- `grid_unverified`, `grid_mismatch`, `partial_view`, `non_planar`, `non_planar_internal` (düzeltilemediyse), `internal_geometry: unmeasurable`, `forced_low_res`, `pile_blur`, `dither_pattern_default`, `class_uncertain`, `class_unsupported`, `class_mixed`, `guard_heavy` → `needs_review`.
- `edit` modunda ek: `edit_drift` (`edit_align_px_p95 > edit_align_max`), `edit_added` (`edit_added_pct > edit_added_max`), `edit_missing` (`edit_missing_pct > edit_missing_max`), `edit_low_res` (`edit_px_per_knot < edit_px_per_knot_min`), `edit_timeout`, `edit_fallback` (arka uç hata verdi, taban çizgisi kullanıldı), `edit_no_gain` (hiçbir aday taban çizgisini geçemedi, taban çizgisi kullanıldı) → `needs_review`.
- `(design_class, design_mode, edit_agent | null, regularize)` dörtlüsü `qc_calib.json.calibrated_configs` içinde yoksa → `mode_uncalibrated`, `needs_review`. Agent dosyasının hash'i de `calibrated_configs`'a yazılır; prompt şablonu değişirse konfigürasyon kalibre edilmemiş sayılır.
- `qc_calib.json` yoksa veya `qc_calib_stale` → `needs_review` (skor hesaplanır, karar verilmez).

**Yumuşak özellikler.** QC modeli az örnekle fit edileceği için özellik seti küçük ve sabittir. Ham ölçümler raporlanır; modele giren set birleştirilmiş **çekirdek set**tir. Öncelik sırası sabittir; kalibrasyonda `n_özellik ≤ n_örnek / 15` kuralı kuyruğu düşürür.

| # | Model özelliği | Tanım | Null kuralı |
|---|---|---|---|
| 1 | `rerender_edge_iou` | `edge_fidelity(rerender, rectified)`: guided filter (r=8) + Lab L Canny, 2 px tolerans kenar IoU. | `edge_density < edge_density_min` |
| 2 | `palette_deltaE_p95` | `rectified` piksellerinin en yakın palet girişine ΔE2000 p95 (gradyan maskesinde dither karışım rengine). | — |
| 3 | `rerender_color_deltaE_p95` | Render ve `rectified`, 16×16 blok Lab ortalamaları arası ΔE2000 p95. | — |
| 4 | `line_jaggedness` | Sınır zincirlerinde 16 düğümlük pencerede run-length entropisi, gradyan maskesi dışı. | `edge_density < edge_density_min` |
| 5 | `noise_1knot_pct` | Temizleme öncesi tek düğümlük bileşenlerin düğüm oranı, gradyan maskesi dışı. Her sınıfta hesaplanır; `distressed`'ta yüksekse çıktı fotoğraf dokusunu taşıyor demektir. | — |
| 6 | `postprocess_changed_pct` | `cleanup_changed_pct + regularize_changed_pct + guard_reverted_pct` (null bileşenler 0 sayılır). | — |
| 7 | `structure_mismatch` | `max(symmetry_consistency, repeat_mismatch_raw)`; ikisi de yoksa null. | simetri ve tekrar yok |
| 8 | `banding` | Gradyan maskesinde geri-render ile `rectified` arasındaki L farkının düşük frekanslı (σ = 8 düğüm) RMS'i. | maske yok |
| 9 | `edge_width_knots` | Kenar geçiş genişliği, düğüm. | — |
| 10 | `illum_residual` | Düzleştirme sonrası kalan aydınlatma gradyanı. | — |

Null değer: standardize edilmiş uzayda 0 (ortalama) atanır; yapısal yokluğu sınıf ve mod one-hot'ları taşır. Kalibrasyon setinde > %50 null olan özellik modelden çıkarılır.

Rapor amaçlı, modele **girmeyen** ölçümler: `rerender_color_deltaE_mean`, `palette_deltaE_p50`, `symmetry_consistency`, `repeat_mismatch_raw`, `noise_knots_pct`, `cleanup_changed_pct`, `regularize_changed_pct`, `guard_reverted_pct`, `wb_confidence`, `illum_range`, `model_uncertainty` (segment: `0.3 < boundary < 0.7` oranı, maske dışı), `edit_align_px` (medyan, p95), `edit_added_pct`, `edit_missing_pct`, `edit_rounds`, `edit_candidate_spread`, `edit_select_margin`. `edit_*` büyüklükleri sert kapı olarak çalışır; modele özellik olarak girmez (edit örneklemi kalibrasyon başına ≤ 20 iş, model kapasitesi yetmez).

**Skor:** ikili lojistik regresyon (`sklearn.linear_model.LogisticRegression`, `penalty="l2"`, `C` iç 5-kat CV ile `{0.1, 0.3, 1, 3}` arasından), hedef `puan ≥ qc_target`. Sayısal özellikler standardize, `design_class` one-hot (≥ 10 örneği olan sınıflar ayrı düzey, kalanlar `other`), `design_mode` one-hot. `qc_target ∈ {5, 4, 3}`, hem ≥ t hem < t sınıfında ≥ 15 örnek olan en yüksek t. `qc_score = P(puan ≥ qc_target)`. `P(puan = 5)` (`qc.p5`) ayrı bir ikili modelden, sadece puan 5 ve puan < 5 sınıflarının ikisinde de ≥ 15 örnek varsa; aksi halde `null`.

**Karar:** `qc_score ≥ qc_accept` → `accepted`; `< qc_reject` → `rejected`; arada → `needs_review`. `qc.status` hedefle yazılır: `accepted@4`. `qc.reasons`: skoru en çok düşüren 3 özellik (katsayı × standardize değer).

**Kalibrasyon protokolü (`eval/calibrate_qc.py`):**
1. ≥ 100 çıktı, sınıf kotası: ≥ 15 `classic`, 15 `geometric`, 10 `gradient`, 10 `distressed`, 5 `minimal`; kota dolmayan sınıf `other`'a düşer ve o sınıfta karar verilmez. Her çıktı sınıfının varsayılan konfigürasyonuyla; kalibre edilen konfigürasyonlar `calibrated_configs`'a yazılır. `edit` konfigürasyonları için ayrı kota: agent yarışını kazanan her agent × sınıf için ≥ 10 çıktı (en fazla iki agent); eşikleri `edit_gate_source: eval` olmayan konfigürasyon kalibrasyona alınmaz. Desenci kör puanlar + süre + tür; `edit` çıktıları diğerleriyle karışık sırada puanlanır, mod desenciye söylenmez.
2. Değerlendirici tutarlılığı: 10 çıktı ≥ 2 hafta sonra ikinci kez; ağırlıklı kappa (`sklearn.metrics.cohen_kappa_score`, quadratic). κ < 0.5 → tüm set iki kez puanlanır, ortalama etiket olur.
3. Çekirdek özellik seti hesaplanır; `n_özellik = min(10, floor(n_örnek / 15))`, öncelik sırasıyla.
4. Model, 5-kat çapraz doğrulama (sınıf katmanlı); AUC toplam, sınıf başına ve mod başına. Toplam < 0.75 → red. Sınıf < 0.65 → `class_unreliable`, o sınıfta `needs_review`. Mod (`edit` + agent) < 0.65 → `config_unreliable`, o konfigürasyon `calibrated_configs`'tan çıkarılır.
5. `qc_accept`: puan ≥ hedef olanları ≥ 0.9 kesinlikle yakalayan en düşük eşik. `qc_reject`: puan ≤ 2 olanların ≥ %90'ını düşüren en yüksek eşik. Profildeki 0.8 / 0.3 kalibrasyona kadar rapor amaçlıdır.
6. Yeniden kalibrasyon: model hash veya agent hash değişince zorunlu; aksi halde her 200 işte 30 rastgele çıktı (sınıf ve mod oranı korunarak), sapma > 0.1 ise yeniden.

**Kapasite (`eval/capacity.py`):** düzeltme süresi sağa çarpıktır, sınıflar arası fark büyüktür; ortalama yanıltır. Raporlanan: sınıf başına, mod başına ve toplamda medyan ve p90 dakika; `iş/gün_tipik = desenci_saat × 60 / medyan`, `iş/gün_kötü = desenci_saat × 60 / p90`; `accepted` oranı sınıf ve mod başına; `edit` için iş başına `cost_usd` medyanı ve GPU süresi.

---

## Lisans ve maliyet tablosu

| Bileşen | Lisans / erişim | Not |
|---|---|---|
| SAM 3.1 (facebookresearch/sam3) | SAM License | Ticari serbest, HF gated (**erişim alındı**), dağıtımda lisans kopyası zorunlu, askeri/ITAR yasak. |
| Qwen-Image-Edit-2511 (Qwen/Qwen-Image-Edit-2511) | Apache 2.0, açık ağırlık | 20B MMDiT + Qwen2.5-VL kodlayıcı; çoklu görsel girdi (fotoğraf + palet kartı); LoRA eğitilebilir; bf16 ~40 GB VRAM, FP8/GGUF ile 24 GB'a sığar; Lightning 4-adım LoRA ile hızlı çıkarım. **`edit` birincil arka ucu.** Eylül 2026 itibarıyla en son düzenleme sürümü; Qwen-Image-2512 metinden görüntü modelidir, düzenleme arka ucu değildir. |
| Qwen-Image-Edit-2509 (Qwen/Qwen-Image-Edit-2509) | Apache 2.0, açık ağırlık | Önceki sürüm; ablasyon arka ucu, opsiyonel. |
| Muse Image 1.0 (Meta) | Ağırlık yok; Meta Model API (`dev.meta.ai`) veya sağlayıcılar (fal, Runware, OpenRouter), ~$0.01/görsel | Akıl yürüten ("agentic") düzenleme modeli; tek görüntü + talimat, maske yok; ≤ 10 referans görsel; önceki çıktıyı geri vererek yinelemeli düzenleme; **seed yok, nondeterministik**; müşteri fotoğrafı dışarı çıkar. `allow_api: true` şart. |
| diffusers, transformers, accelerate | Apache 2.0 | Qwen düzenleme arka ucu (`QwenImageEditPlusPipeline`) |
| peft | Apache 2.0 | opsiyonel LoRA |
| timm | Apache 2.0 | `segment` encoder (convnext_tiny) |
| segmentation-models-pytorch | MIT | yedek dekoder |
| opencv-contrib-python | Apache 2.0 | ximgproc, ECC, TPS |
| scikit-image, scipy, numpy, tifffile | BSD | |
| scikit-learn | BSD | k-means, QC lojistik regresyon, çapraz doğrulama, kappa |
| pyyaml | MIT | |
| noise | MIT | simplex gürültü |
| ultralytics | AGPL | KULLANMA |

---

## Girdi

| Alan | Kaynak | Zorunlu |
|---|---|---|
| Fotoğraf — önerilen uzun kenar ≥ `photo_long_px_rec` (3000 px); JPEG kalite ≥ 90; sert kapı efektif çözünürlüktür (`rectify` adım 12) | müşteri | evet |
| `width_cm`, `height_cm` — hav alanı, saçak hariç | müşteri | evet |
| `density` — anlamı `density_definition` ile | müşteri / fabrika | evet |
| `machine_colors` (5/6/8/12) | fabrika | evet |
| `consent_train: true/false` | müşteri | evet |
| `design_class` — sınıf adı; `auto` sadece sentetik/toplu değerlendirmede | desenci | **evet** (gerçek işte) |
| Tezgah profili, sınıf dosyaları, agent dosyaları | config | evet |
| İplik kütüphanesi | `config/yarns/*.csv` | hayır |
| `design_mode` (`segment | quantize | edit`; sınıf varsayılanını ezer) | desenci | hayır |
| `edit_agent` — `design_mode: edit` ise zorunlu; `config/agents/<ad>.yaml` | desenci | koşullu |
| `edit_seed` — `qwen_edit_*` için tohum; profil varsayılanını ezer | desenci | hayır |
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
    ├── 05_design/baseline/  baseline_rgb.png, ref_indexed.npy   (always in edit mode; cached separately)
    └── 05_design/edit/      round_<r>/cand_<k>.png, cand_<k>_aligned.png, cand_<k>_features.json, prompt_<r>.txt, select.json
```

`report.json`:
```json
{
  "versions": {"git_sha": "...", "profile_hash": "...", "class_config_hash": "...", "yarn_csv_hash": "...",
               "sam_checkpoint": "facebook/sam3.1", "segment_ckpt_sha256": null, "qc_calib_hash": "...",
               "edit_agent_hash": "...", "edit_backend_version": "Qwen/Qwen-Image-Edit-2511@<rev>", "edit_lora_sha256": null},
  "setup": {"reed_per_m": "profile", "density_definition": "knots_m2", "dither_pattern": "archive",
            "yarn_library": false, "motif_min_knots": "archive", "noise_scale": "assumed", "qc_calib": true,
            "sam": true, "segment_ckpt": false,
            "edit_backends": {"qwen_edit_2511": true, "qwen_edit_2509": false, "muse_image": true}},
  "photo": {"native_long_px": 1920, "long_px_below_rec": true},
  "grid": {"W": 840, "H": 1140, "reed_per_m": 420, "wefts_per_m": 380, "grid_verified": true,
           "photo_px_per_knot": 1.7, "edge_width_knots": 1.1, "photo_px_per_knot_eff": 1.55,
           "knot_grid": {"visible": false, "spacing_px": null, "mismatch_pct": null},
           "upsampled": true},
  "design_class": "gradient", "class_source": "designer",
  "class_auto": "gradient", "class_margin": 0.52, "class_conflict": false, "class_mixed": false,
  "class_features": {"edge_density": 0.04, "n_colors_est": 6, "gradient_frac": 0.38,
                     "noise_frac": 0.01, "noise_1knot_frac": 0.03, "line_frac": 0.2, "thin_line_frac": 0.002},
  "colors": {"machine_colors": 12, "palette_used": 9},
  "fringe": {"detected": [true, true, false, false], "strip_px": [42, 39, 0, 0]},
  "illumination": {"range_L": 11.4, "residual": 0.8, "flattened": true},
  "internal_geometry": {"source": ["border", "ground_repeat"], "tps_applied": true,
                        "max_dev_knots_before": 2.3, "max_dev_knots_after": 0.4},
  "palette_confirmed": false,
  "design_mode": "edit", "design_mode_requested": "edit", "design_mode_class_default": "quantize",
  "edit": {"agent": "qwen2511_loop", "backend": "qwen_edit_2511", "seed": 0,
           "px_per_knot": 2.0, "tiled": true, "n_tiles": 6,
           "rounds": 2, "n_candidates": 3,
           "align_px": {"median": 1.8, "p95": 4.1}, "added_pct": 0.004, "missing_pct": 0.002,
           "gate_source": "eval",
           "baseline_edge_iou": 0.41, "selected_edge_iou": 0.55,
           "candidate_spread": 0.05, "select_margin": 0.12, "selected": "round_1/cand_2",
           "wall_s": 412.0, "cost_usd": 0.0, "api_egress": false},
  "regularize": {"applied": true, "changed_pct": 0.014, "line_jaggedness_before": 0.41,
                 "line_jaggedness_after": 0.12, "repeat_mismatch_raw": 0.06,
                 "segment_classes": {"line": 0.31, "arc": 0.12, "free": 0.57, "free_len_share": 0.62}},
  "guard": {"reverted_components": 2, "reverted_area_pct": 0.0003,
            "by_stage": {"production_cleanup": 0, "production_merge": 1, "regularize_repeat": 0, "regularize_smooth": 1}},
  "dither": {"applied": true, "mask_pct": 0.38, "levels": 4, "pairs": [[2, 5], [5, 7]]},
  "qc": {"status": "needs_review@4", "score": 0.61, "p5": null, "target": 4,
         "features": {"rerender_edge_iou": 0.55, "palette_deltaE_p95": 14.1, "rerender_color_deltaE_p95": 6.2,
                      "line_jaggedness": 0.12, "noise_1knot_pct": 0.02, "postprocess_changed_pct": 0.019,
                      "structure_mismatch": null, "banding": 1.1},
         "hard_gate_failures": [],
         "reasons": ["palette_deltaE_p95=14.1", "rerender_edge_iou=0.55", "banding=1.1"]},
  "timings_s": {"rectify": 3.1, "structure": 2.0, "design": 412.0, "regularize": 4.2, "dither": 0.6, "qc": 2.4, "total": 425.0},
  "warnings": ["photo_long_px_below_rec", "low_photo_resolution"],
  "partial_view": false, "non_planar": false, "non_planar_internal": false,
  "deterministic": false
}
```

- `design_mode ≠ design_mode_class_default` → desenci ezmiş, yönlendirme kuralı çalışmış veya arka uç hata vermiştir; `warnings` nedenini yazar (`segment_fallback`, `edit_thin_lines`, `edit_fallback`, `edit_no_gain`).
- `edit` bloğu `design_mode ≠ edit` ise `null`.
- `palette_confirmed`: `rerun --from production --palette ...` veya `--confirm-palette` ile `true`.
- `deterministic`: `segment`/`quantize` → `true`; `edit` + `qwen_edit_*` → `true` sadece aynı seed, aynı ağırlık, aynı donanım sınıfında tolerans içinde (`test_edit.py`); `edit` + `muse_image` → `false`.

Saklama: `consent_train: false` → `retention_days` sonra silinir; `true` → `data/real/pairs/`. `edit` ara adayları (`stages/05_design/edit/`) `consent_train`'den bağımsız `retention_days` sonra silinir; seçilen aday `design_rgb.png` olarak kalır.

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
- `edit` + `qwen_edit_*`: aday `k`'nin seed'i `edit_seed + 1000·r + 10·k + t` (`t` karo indeksi, karosuz 0); aynı seed aynı donanım sınıfında tolerans içinde tekrarlanabilir. Aday seçimi skoru eşit adaylarda en düşük `(r, k)` kazanır.
- `edit` + `muse_image`: tekrarlanamaz. Cache aday görsellerini saklar; `rerun --from production` ve sonrası deterministiktir, `rerun --from design` API'yi yeniden çağırır ve farklı sonuç verebilir.

---
## Aşamalar

Sıra: `intake → rectify → structure → palette → design → production → regularize → dither → qc → export`. Her aşama tek girdi → tek çıktı; girdi hash'i (çözülmüş parametre seti dahil) değişmediyse yeniden çalışmaz. `rerun --from <ad>` o aşama ve sonrasını çalıştırır.

Çalışma çözünürlüğü tek ve sabittir: **4 px/düğüm**, `rectified.png` tam olarak `4W × 4H`, düğüm ızgarası piksel ızgarasına kilitli. `edit` modu bu kuralın tek istisnasıdır ve istisna `design` aşamasının içinde kalır: arka uç kendi çözünürlüğünde çalışır, aşamanın çıktısı yine `4W × 4H`'dir.

### intake

Girdi: fotoğraf, iş parametreleri, profil, sınıf dosyaları, agent dosyası (koşullu), iplik CSV (opsiyonel)
1. Doğrulama: dosya; `width_cm`/`height_cm` > 0; `density` > 0; `machine_colors ∈ allowed_machine_colors`; `consent_train` verilmiş; `design_class` verilmiş ve `config/classes/<class>.yaml` var, ya da `auto` (`--class auto` sadece `synthetic`/`batch` bayrağıyla kabul edilir; gerçek işte hata). `design_mode: edit` ise `edit_agent` verilmiş ve `config/agents/<agent>.yaml` var; agent'ın `backend`'i `muse_image` ise profilde `allow_api: true`, değilse `api_not_allowed` hata. Override alanları doğrulanır.
2. EXIF döndürme. `photo.native_long_px`; `< photo_long_px_rec` → `warnings += "photo_long_px_below_rec"` (bilgi; kapı `rectify` adım 12). Uzun kenar üst sınırı `max(4096, 4 × max(W, H))`.
3. Izgara. `reed_per_m` (yatay) ve `wefts_per_m` (dikey). `density_definition`:
   - `knots_m2`: `wefts_per_m = density / reed_per_m`.
   - `wefts_per_m`: `wefts_per_m = density`.
   - `reed_x_wefts`: `density = [tarak, atkı]`; `density[0] ≠ reed_per_m` → `grid_input_mismatch` hata.
   - `W = round(width_cm / 100 × reed_per_m)`, `H = round(height_cm / 100 × wefts_per_m)`; yuvarlama hatası > 0.5 düğüm → `grid_rounding`.
   - `reed_per_m: null`: sadece `knots_m2` ile, `k = sqrt(density)`; `grid_unverified`. `structure` gerçek tarağı ölçer (`grid.measured_reed_per_m`).
4. Palet üst sınırı `machine_colors`.
5. İplik kütüphanesi varsa CSV → `yarn_lab.npy`.
6. Parametre katmanı ilk çözüm (sınıf verilmişse nihai; `auto` ise profil değerleriyle, `structure` sonrası yeniden). `edit` ise agent katmanı dahil. Yönlendirme kuralları (`segment_fallback`, `edit_thin_lines`) `structure` sonunda uygulanır.
7. `versions`, `setup`, `photo` → `job.json`.
Çıktı: `job.json`, `photo.png`, `yarn_lab.npy` (opsiyonel)

### rectify

Girdi: `photo.png`, `job.json`
1. Köşe kaynağı: `corners_override` varsa maske = dörtgen, adım 5'e. Yoksa SAM 3.1 (`models/sam3.1/`), prompt `"carpet"` → `"rug"` → `"carpet on the floor"` (skor < 0.3'te sıradaki), seçim `score × alan`. Checkpoint yüklenemezse `sam_unavailable`, `needs_corners` ile dur.
2. En büyük bağlı bileşen, delik doldurma.
3. Kaba dörtgen: maske üzerinde `minAreaRect` → 4 kaba köşe → kaba homografi ile maske ve `photo` düzleştirilir (sadece bu adım için, geçici).
4. Saçak tespiti: kaba düzleştirilmiş görüntüde 4 kenar için 1-B profil (doygunluk ortalaması, 5 px yerel varyans). Düşük doygunluk + yüksek varyans şerit, ardından keskin geçiş (CUSUM). Genişlik kenarın %2–`fringe_max_frac`'ı arasında ve kısa kenarlardan en az birinde varsa kabul; şerit kaba homografinin tersiyle orijinal maskeye geri taşınır ve maske geçişe kırpılır. Belirsizse `fringe_uncertain`. Uzun kenarda `fringe_long_side`.
5. Kısmi kadraj: maske görüntü kenarına 3 px'ten yakın ve temas çevrenin > %10'u → `partial_view`. `allow_partial: false` ise dur.
6. Köşe (kırpılmış maske üzerinde): `approxPolyDP` (eps = çevre × 0.02) → 4 köşe; olmazsa `HoughLinesP` kesişimleri; olmazsa adım 3'ün `minAreaRect` köşeleri + `corners_weak`.
7. Düzlemsellik (dış): kontur–dörtgen en büyük dik uzaklık / kenar → `planarity_dev`; `> planarity_warn` → `non_planar`; `> planarity_stop` → dur.
8. Geometrik çözünürlük: `photo_px_per_knot = min(4 kenar için kaynak px / düğüm)`. `upsampled = photo_px_per_knot < 4`.
9. Homografi: 4 köşe → `(0,0), (4W,0), (4W,4H), (0,4H)`. INTER_AREA küçültme / INTER_CUBIC büyütme.
10. Gray-world beyaz dengesi → `wb_gains`, `wb_clipped_pct`.
11. Aydınlatma düzleştirme: Lab L, log uzayı; maske = guided filter + Canny kenarlarının 3 px dilate'i dışı; 2. derece iki değişkenli polinom, Huber. Ortalama L korunur. `illum_range > illum_warn` → `strong_illumination`. a/b düzleştirilmez. `rectified.png` (renk-doğru) ve `rectified_contrast.png` (L CLAHE clip 2.0 tile 8; sadece `structure` ve kenar işlemleri).
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
   - `rectified` INTER_AREA ile **W×H'ye** (ızgara çözünürlüğü); her düğüm en yakın merkeze atanır (yeniden fit yok) → `coarse_labels.npy` (H×W). `noise_frac`, `noise_1knot_frac`, `thin_line_frac` ve adım 9'un bileşenleri bu haritadan.
   - Sınıf tespiti (bkz. Desen sınıfı). `class_source: designer` ise sadece rapor + `class_conflict`. `class_mixed` her durumda değerlendirilir.
   - `auto` ise parametre katmanı yeniden çözülür. Ardından yönlendirme kuralları (`segment_fallback`, `edit_thin_lines`); nihai `design_mode` `job.json`'a yazılır.
7. İç geometri ölçümü, düğüm cinsinden sapma, üç kaynak:
   - `border`: iç kenar dizisi doğrusal regresyona uydurulur, maksimum sapma.
   - `ground_repeat`: `lattice_nodes` ile ideal kafes arası sapma.
   - `knot_grid`: `photo_px_per_knot_eff ≥ 2` ve `pile_blur` yoksa `rectified_contrast` tam çözünürlükte 64×64 px bloklarda 2-B otokorelasyon; beklenen tepeler (4,0), (0,4) px; tepe ≥ 0.4 güvenli blok oranı ≥ 0.7 → `visible`. (a) Global medyan aralık; `|spacing − 4| / 4 > grid_mismatch_tol` → `grid_mismatch` (sert kapı). Gerçek tarak → `measured_reed_per_m`. (b) Blok bazında sapmadan yer değiştirme alanı → `internal_dev_knots`, TPS kontrol noktaları.
   `internal_dev_knots` = maksimum. `> planarity_warn × kenar` → `non_planar_internal`. Üç kaynak da yoksa `internal_geometry: unmeasurable` (sert kapı).
8. TPS düzeltmesi (`non_planar_internal` ise): kontrol noktaları bordür iç kenarlarında 16'şar nokta, `lattice_nodes`, `knot_grid` alanından 8×8 örnek, simetri ekseninde 8 nokta, dış 4 köşe sabit. `cv2.createThinPlateSplineShapeTransformer` ile `rectified*.png` warp (4W × 4H korunur). Doğrulama: adım 7 tekrar; `internal_dev_knots` ≥ %50 düşmediyse geri alınır, `tps_no_gain`. Düştüyse `tps_applied: true`; `coarse_labels.npy` yeniden üretilir.
9. Bölge haritası, **ızgara çözünürlüğünde**: 512 px'te tespit edilen bölge maskesi (zemin/bordür/madalyon/köşe) ızgaraya yukarı örneklenir; `coarse_labels.npy` üzerindeki bağlı bileşenler **bütün olarak** en çok örtüştüğü bölgeye atanır → `regions.npy` (H×W, uint8): 0 = ground, 1 = border, 2 = medallion, 3 = other, 4 = corner. Tek tüketici `regularize`.
Çıktı: `layout.json`, `regions.npy`, `coarse_labels.npy`, `gradient_mask.npy`, `gradient_regions.json`, güncellenmiş `rectified*.png` (TPS uygulandıysa), `job.json` (`resolved_params`, nihai `design_mode`)

Zorlama kararı (`regularize`'da): `force_*` — `auto` → ilgili güven ≥ `enforce_conf` **ve** `design_class ∈ {classic, geometric}`; `on` → her durumda (tespit yoksa uyarı ile atlanır); `off` → asla. Zorlama guard'dan geçer. Tespit eşiği (0.6) ile zorlama eşiği (0.85) ayrı.

### palette

Girdi: `rectified.png`, `gradient_mask.npy`, `gradient_regions.json`, `job.json`, `yarn_lab.npy` (opsiyonel), `palette_override`
Palet modelden **önce**, fotoğraftan seçilir. Üst sınır `machine_colors`, alt sınır 1. `edit` modunda palet ayrıca prompt'a ve palet kartı görseline girer.
1. `rectified` → guided filter (r=8, eps=0.01) → Lab → uzun kenar 400 px → k-means (k = 4 × `machine_colors`, sabit seed). Küme ağırlığı `count^area_weight_exp`. Gradyan maskesi içindeki pikseller ağırlık 0; her gradyan bölgesinin iki uç Lab değeri zorunlu aday.
2. Komşuluk matrisi: 400 px'te iki kümenin paylaştığı sınır piksel sayısı; gradyan maskesi içindeki sınırlar sayılmaz.
3. Seçim:
   - `palette_override` varsa: verilen iplikler.
   - Kütüphane varsa: aday = tüm iplikler; açgözlü seçim, her adımda ağırlıklı toplam ΔE2000'i en çok düşüren iplik; kazanç < `gain_stop_select` (ΔE birimi) → dur. Gradyan uçları için en yakın iplik zorunlu.
   - Kütüphane yoksa: açgözlü küme birleştirme, maliyet = ağırlıklı ΔE artışı + `adjacency_penalty × paylaşılan sınır / toplam sınır`. `machine_colors`'a in; ardından birleştirme maliyeti `merge_cost_stop` altında kaldığı sürece birleştirmeye devam et. Gradyan uç kümeleri birleştirilmez.
   - İki durdurma eşiği ayrıdır: biri kazanç (ΔE, pozitif yönde), diğeri maliyet (ΔE + ceza).
4. Motif rengi koruması: kapsama küçük ama bileşen sayısı ≥ `motif_color_min_cc` veya en büyük bileşeni ≥ `motif_min_knots` olan kümeler birleştirmede son sırada.
5. `palette.json` (indeks, RGB, Lab, `yarn_code`, `gradient_endpoint`), `palette_proposed.json`, `palette_card.png` (her palet girişi için 128×128 px düz kare, altında hex kodu; `edit` referans görseli).
Çıktı: `palette.json`, `palette_proposed.json`, `palette_card.png`

### design

Girdi: `rectified.png`, `gradient_mask.npy`, `coarse_labels.npy`, `palette.json`, `palette_card.png`, `job.json`, çözülmüş parametreler
Simetri/tekrar zorlama burada yapılmaz (`regularize`). Üç modun ortak çıktısı `design_rgb.png` (`4W × 4H`); `segment` ek olarak `design_lab.npy` + `boundary.npy` üretir.

**quantize (taban çizgisi, profil varsayılanı, `distressed` varsayılanı, yedek)**
1. `rectified.png` 4 px/düğüm'de guided filter (r=8, eps=0.01), mean-shift (sp=24, sr=20).
Çıktı: `design_rgb.png`
`distressed` gerekçesi: mean-shift gürültüyü olduğu gibi bırakır; `segment`'in sınır kanalı eskitme gürültüsünde doyar ve L1(Lab) gürültünün ortalamasını öğrenir.

**segment (eğitim ve eval sonrası classic, geometric, gradient, minimal varsayılanı)**
Deterministik. Motif uydurmaz; girdi belirsizse bulanık çıktı verir, `production` en yakın ipliğe kuantize eder, hata görünür kalır.
1. Model `SegNet` (`design_seg.py`): `timm.create_model("convnext_tiny", features_only=True, out_indices=(0,1,2,3))` + FPN dekoder, stride 4 → düğüm başına 4 kanal: Lab (3, regresyon) + sınır olasılığı (1, sigmoid). Yedek: `smp.Unet(convnext_tiny)` + `avg_pool2d(4)`.
2. Karo çıkarımı: 512×512 px (128×128 düğüm), 4 px katı, örtüşme 64 px, merkez 448×448 alınır.
3. Gradyan bölgede istenen davranış: düz (yerel ortalama) Lab + düşük sınır olasılığı; eğitim hedefi buna göre kurulur. Ara renkleri `dither` üretir.
4. Çıkarım hata verirse `design_mode → quantize`, `warnings += "segment_fallback"` (checkpoint yokluğu `structure` sonunda yakalanır).
Çıktı: `design_lab.npy` (H×W×3), `boundary.npy` (H×W), `design_rgb.png`

**edit (agent'lı diffusion düzenleme)**
Bir agent = arka uç + prompt politikası + aday sayısı + tur sayısı + seçim ağırlıkları; hepsi `config/agents/<ad>.yaml`'da. Döngü `agents/loop.py`'de ortaktır, arka uçlar `agents/backends/`'de. Eleştirmen (critic) **ölçümlerdir**, dil modeli değil: her aday hizalanır, kuantize edilir ve QC özelliklerinin hızlı alt kümesi hesaplanır; geri bildirim prompt'u bu sayılardan kurallarla üretilir.

0. Taban çizgisi (her edit işinde, önce): `quantize` yolu → `baseline_rgb.png`; hızlı kuantizasyon (adım 3d) → `ref_indexed.npy` (H×W). `stages/05_design/baseline/`'da ayrı cache anahtarıyla saklanır (`edit_*` parametrelerinden bağımsız; agent değişince yeniden hesaplanmaz). Üç işlevi vardır: aday özelliklerinin referansı, `edit_fallback` / `edit_no_gain` durumunda çıktı, `baseline_edge_iou = edge_fidelity(baseline_rgb, rectified)` kıyas değeri.
1. Çalışma çözünürlüğü: tek geçiş `edit_px_per_knot = min(edit_px_per_knot_target (2.0), backend.max_long_px / max(W, H))`. `edit_tiling: auto` ve tek geçiş `< edit_px_per_knot_target` ise karolama: `edit_px_per_knot = edit_px_per_knot_target`, karo `edit_tile_px` (1024) × `edit_tile_px`, örtüşme 128 px, her karo aynı prompt + palet kartı ile ayrı düzenlenir, örtüşmede doğrusal harmanlama; `tiled: true`, `n_tiles`. `edit_tiling: off` ise tek geçiş; `edit_px_per_knot < edit_px_per_knot_min` (1.0) → `edit_low_res` (sert kapı). `rectified.png` INTER_AREA ile hedef çözünürlüğe küçültülür, arka ucun 16 px katına yuvarlanır (`edit_input.png`). Karolu düzenleme dikiş ve karo başına farklı yorum riski taşır; dikiş hatası `edit_align_px` blok istatistiklerinde görünür, karolu/karosuz karşılaştırma agent yarışında yapılır.
2. Prompt: `prompts.py` şablonu. Taban (İngilizce):
   ```
   Redraw this carpet photo as its flat production design: a top-down, perfectly rectangular
   drawing of the same pattern. Hard, clean edges. No pile texture, no shading, no shadows,
   no highlights, no fringe, no perspective. Use ONLY the colors in the second image
   (hex: #C8B798, #3A2A1E, ...). Keep every motif at exactly the same position, size and
   shape as in the photo. Do not add, remove, complete, or reinterpret any motif.
   ```
   Sınıf ekleri: `gradient` → "Where the photo shows a smooth color transition, keep it smooth; do not invent steps."; `distressed` → "Keep the worn, speckled areas as they are; do not clean them up."; `geometric` → "Lines must be perfectly straight; repeated motifs must be identical."; `minimal` → "Large flat regions; no texture inside them."
   Referanslar: `edit_input.png` (görsel 1), `palette_card.png` (görsel 2). `qwen_edit_2511` çoklu görsel girişi destekler; `muse_image` ≤ 10 referans alır. Negatif prompt (`qwen_edit_*`): `"photo, texture, fabric, shadow, blur, noise, watermark, text"`.
3. Tur `r = 0 … edit_max_rounds − 1`:
   a. `edit_n_candidates` aday üretilir. `qwen_edit_*`: seed formülü Determinizm bölümünde; `num_inference_steps` ve `true_cfg_scale` agent dosyasından; Lightning LoRA açıksa 4 adım. `muse_image`: aynı prompt `n` kez (seed yok), `resolution: 2K`. `r > 0`'da girdi görseli önceki turun seçilen adayıdır (yinelemeli düzenleme), fotoğraf ise ek referans olarak verilir (görsel 3) ve prompt'a "Reference photo is image 3; correct the drawing (image 1) to match it." eklenir.
   b. Her aday: arka uç çıktısı (karolu ise harmanlanmış) → `4W × 4H`'ye INTER_CUBIC (`cand_<k>.png`).
   c. Hizalama (`knotvision/align.py`, `train/align.py` ile aynı kod): aday ↔ `rectified`, Lab L, global ECC homografi (3 seviye piramit) + 4×4 blok yerel affine → aday warp edilir (`cand_<k>_aligned.png`), blok yer değiştirme medyanı ve p95 → `align_px`. ECC yakınsamazsa aday `align_px = inf` ile elenir.
   d. Hızlı kuantizasyon: `production` adım 2'nin `quantize` yolu (en yakın giriş + 4×4 mode pooling) → geçici `indexed_cand`. Temizleme ve guard çalışmaz.
   e. Özellikler (`cand_<k>_features.json`): `edge_fidelity(cand_aligned, rectified)` (→ `rerender_edge_iou` eşdeğeri), `palette_deltaE_p95` (adayın palet dışına çıkma miktarı), `edit_added_pct`, `edit_missing_pct` (`indexed_cand` ↔ `ref_indexed`: aynı indeks, IoU < 0.3, alanı > `motif_min_knots`; gradyan maskesi dışı), `noise_1knot_pct`, `line_jaggedness`, `align_px`.
   f. Seçim skoru `s = Σ w_i · z_i` (ağırlıklar agent dosyasında, `z` = profil sabitleriyle ölçeklenmiş özellik; `edge_iou` ve düşük `added/missing/align/deltaE` yüksek skor). Tek seçim ölçütü budur; kalibre QC modeli aday özellik setiyle fit edilmediği için aday seçiminde kullanılmaz.
   g. Durma: en iyi aday tüm `edit_pass_*` eşiklerini geçiyorsa (`align_px_p95 ≤ edit_align_max`, `added_pct ≤ edit_added_max`, `missing_pct ≤ edit_missing_max`, `palette_deltaE_p95 ≤ edit_palette_deltaE_max`) → dur. Değilse ve `r < edit_max_rounds − 1` → geri bildirim.
   h. Geri bildirim (kurallar; ihlal eden her özellik için bir cümle prompt'a eklenir):
      - `align_px` yüksek → "Do not move, shift, rotate or rescale anything; every motif stays exactly where it is."
      - `added_pct` yüksek → "Remove any element that is not in the reference photo. Do not complete faint or partial motifs."
      - `missing_pct` yüksek → "Some motifs from the photo are missing; draw every motif visible in the photo, including faint ones."
      - `palette_deltaE_p95` yüksek → "Use only the exact colors from the palette image; no intermediate tones, no gradients unless present in the photo."
      - `noise_1knot_pct` yüksek → "Fill each region with one flat color; no speckle, grain or texture."
      - `line_jaggedness` yüksek (`geometric`/`classic` bordür) → "Straight lines must be perfectly straight and parallel to the edges."
      Geri bildirim `prompt_<r+1>.txt`'e yazılır.
   i. Duvar süresi `edit_max_wall_s`'i aşarsa döngü kesilir, eldeki en iyi aday kullanılır, `edit_timeout`.
4. Kazanç kontrolü: seçilen adayın `edge_fidelity` değeri `baseline_edge_iou − edit_gain_tol` (0.02) altındaysa aday atılır, `design_rgb.png = baseline_rgb.png`, `warnings += "edit_no_gain"` (sert kapı). Gerekçe: taban çizgisinden kötü bir diffusion çıktısı için GPU süresi harcanmış demektir; iş desenciye taban çizgisiyle gider, yarış tablosunda `edit_no_gain_rate` görünür.
5. Seçilen aday (`select.json`: tur, aday, skorlar, `select_margin`, `candidate_spread`) → `design_rgb.png` = `cand_aligned`; `design_lab_smooth.npy` = `design_rgb` guided filter (r=8) Lab (dither girdisi). Rapor `edit` bloğu.
6. Arka uç hatası (ağırlık yüklenemedi, VRAM, API 4xx/5xx üç denemeden sonra) → `design_rgb.png = baseline_rgb.png`, `design_mode → quantize`, `warnings += "edit_fallback"`, sert kapı. Ek hesap yoktur; taban çizgisi zaten hazırdır.
7. Maliyet: `muse_image` → `cost_usd = n_görsel × muse_price_usd` (profil, 0.01; karolu işte karo sayısı çarpar); `qwen_edit_*` → 0 (GPU süresi `wall_s`'te). `api_egress: true` yalnız `muse_image`.
Çıktı: `design_rgb.png`, `design_lab_smooth.npy`, `stages/05_design/baseline/`, `stages/05_design/edit/`

Hazır agent dosyaları (`config/agents/`):
```yaml
# config/agents/qwen2511_direct.yaml  -- single pass, no loop; the cheapest edit baseline
backend: qwen_edit_2511
edit_n_candidates: 1
edit_max_rounds: 1
select_weights: {edge_iou: 1.0, added_pct: -2.0, missing_pct: -2.0, align_px: -1.0, palette_deltaE_p95: -0.5, noise_1knot_pct: -0.5, line_jaggedness: -0.3}
num_inference_steps: 30
true_cfg_scale: 4.0
lightning_lora: false
edit_lora: null                 # optional path under models/edit_lora/
```
```yaml
# config/agents/qwen2511_loop.yaml  -- candidates + metric feedback, up to 3 rounds
backend: qwen_edit_2511
edit_n_candidates: 3
edit_max_rounds: 3
select_weights: {edge_iou: 1.0, added_pct: -2.0, missing_pct: -2.0, align_px: -1.0, palette_deltaE_p95: -0.5, noise_1knot_pct: -0.5, line_jaggedness: -0.3}
num_inference_steps: 30
true_cfg_scale: 4.0
lightning_lora: false
edit_lora: null
```
```yaml
# config/agents/muse_loop.yaml  -- API backend; nondeterministic; costs money and sends the photo out
backend: muse_image
edit_n_candidates: 2
edit_max_rounds: 3
select_weights: {edge_iou: 1.0, added_pct: -2.0, missing_pct: -2.0, align_px: -1.0, palette_deltaE_p95: -0.5, noise_1knot_pct: -0.5, line_jaggedness: -0.3}
resolution: 2K
```
`qwen2509_loop.yaml` = `qwen2511_loop` ile aynı, `backend: qwen_edit_2509` (ablasyon; opsiyonel). Yeni agent eklemek = yeni dosya; kod değişmez. Her agent dosyasının hash'i `versions.edit_agent_hash`.

### production

Girdi: `design_rgb.png` veya `design_lab.npy` + `boundary.npy`, `design_lab_smooth.npy` (edit/gradyan), `palette.json`, `gradient_mask.npy`, `ref_indexed.npy` (edit), `job.json`, çözülmüş parametreler
1. Izgara W×H.
2. Kuantizasyon:
   - `quantize` ve `edit`: her piksel → en yakın giriş (ΔE2000); 4×4 mode pooling.
   - `segment`: iki geçiş. Geçiş 1: `boundary ≤ boundary_thr` → en yakın giriş. Geçiş 2: sınır düğümleri → `boundary_snap_r` yarıçapındaki geçiş-1 girişlerinden tahmin Lab'ına en yakını. Gradyan maskesinde iki geçiş uygulanmaz: tahmin Lab'ı `design_lab_smooth.npy` (dither girdisi), indeks geçici olarak en yakın giriş.
3. Gürültü ölçümü (temizleme öncesi): `noise_knots_pct` (alanı ≤ `min_region_knots`), `noise_1knot_pct` (alanı = 1), gradyan maskesi dışı. `edit` modunda ek: `edit_added_pct`, `edit_missing_pct` (`indexed_raw` ↔ `ref_indexed`, temizleme öncesi, nihai değerler; `design`'daki aday değerleri seçim içindir). Bu iki değer `qc`'de sert kapıdır.
4. Bölge temizleme (3 tur): alanı ≤ `min_region_knots` bileşen → en uzun sınırı paylaştığı komşuya. `segment`'te `boundary > 0.5` düğümler muaf; gradyan maskesi muaf; `min_region_knots: 0` → atlanır. `cleanup_changed_pct`. **Guard.**
5. `is_thin(bileşen)`: iskelet uzunluğu ≥ `line_min_knots` VE maksimum yerel genişlik ≤ `line_max_width`. Aynı fonksiyon `structure`'da `thin_line_frac` için ve `tools/motif_stats.py`'de kullanılır (`metrics.py`).
6. Kapsama < `min_coverage_pct` renkleri en yakın girişe birleştir. Muaf: `is_thin` bileşenlerden oluşan renkler; motif rengi kriteri; `gradient_endpoint`. `min_coverage_pct: 0` → atlanır. **Guard.**
7. Rapor: giriş başına `quantization_deltaE`; `palette_deltaE` p50/p95; `palette_model_mismatch` (p95 > `palette_model_mismatch_warn`); `quantization_deltaE > deltaE_warn` olan girişler `warnings += "palette_far"` (bilgi).
Çıktı: `indexed_raw.npy`, `design_lab_smooth.npy` (gradyan varsa), `production_stats.json`, `stages/guard.json` (eklenir)

### regularize

Girdi: `indexed_raw.npy`, `layout.json`, `regions.npy`, `gradient_mask.npy`, çözülmüş parametreler
Deterministik; model yok. `regularize: off` → `indexed = indexed_raw`, istatistikler yine hesaplanır. Gradyan maskesi ve `regions == 4` tüm adımlarda dokunulmaz. Moddan bağımsızdır; `edit` çıktısı da buradan geçer.
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
Gradyan maskesi yoksa boş geçer (`dither.applied: false`). Deterministik, analitik. Moddan bağımsız.
1. Her gradyan bölgesi için rampa: iki uç indeks (`gradient_endpoint`); arada Lab'da sıralı ara indeksler varsa çok segmentli `[i0, i1, ..., ik]`.
2. Her düğümün `design_lab_smooth` değeri rampaya izdüşürülür → komşu iki indeks `(a, b)`, oran `t`.
3. `t` `levels` kademeye yuvarlanır; ordered matris (`config/dither/<name>.yaml`: `type: ordered | row_scan | diagonal_scan`, `matrix`, `levels`) ile `a` veya `b`.
4. Maske kenarındaki düğümler en yakın uç indekse (dither düz alana taşmaz).
5. Rapor: `mask_pct`, `pairs`, `levels`. `dither_pattern: null` → `dither_pattern_default` (sert kapı).
Çıktı: `indexed.npy`

### qc

Girdi: `indexed.npy`, `indexed_raw.npy`, `palette.json`, `rectified.png`, `layout.json`, `gradient_mask.npy`, `production_stats.json`, `regularize_stats.json`, `stages/guard.json`, `stages/05_design/edit/select.json` (edit), `job.json`, `models/qc_calib.json`
1. Sert kapılar (mod bağımlı `edit_*` kapıları dahil; `edit_added` / `edit_missing` `production_stats.json`'daki nihai değerlerle).
2. Geri-render: `indexed` → palet RGB → 2 px/düğüm; sentetik zincirin hav dokusu (aydınlatma, perspektif, kamera yok) → `rerender.png`. `rectified` aynı boyuta.
3. Ham ölçümler; çekirdek özellikler birleştirilir; null kuralları.
4. `qc_calib.json` varsa ve hash uyuşuyorsa skor + `p5` + karar + `reasons`; `class_unreliable` / `config_unreliable` → `needs_review`; yoksa `needs_review`, `score = null`. Hash uyuşmazlığı → `qc_calib_stale`.
5. `preview.png` panelleri; `edit` modunda "model output" paneli seçilen hizalanmış adaydır (veya taban çizgisi, `edit_no_gain` / `edit_fallback` ise), altına tur/aday etiketi yazılır.
Çıktı: `qc.json`, `rerender.png`. Süre < 5 s.

### export

1. `design.tiff`: `tifffile`, `photometric='palette'`, colormap = palet RGB (uint16), `resolution` düğüm/inç (yatay = reed, dikey = atkı).
2. `palette.csv`: `index, yarn_code, name, R, G, B, knots, coverage_pct, quantization_deltaE`.
3. `preview.png`, `report.json`.
4. `consent_train: true` → `data/real/pairs/<job_id>/{photo.jpg, design.tiff, indexed_raw.npy, corners.json, meta.json}`, `reviewed=false`. Desenci nihai TIFF'i → `truth.tiff`, `reviewed=true`. QC özellikleri, puan, süre, tür, sınıf, mod, agent → `meta.json`.
Çıktı: `output/`

---
## Tezgah profili

```yaml
# config/looms/wilton_default.yaml
name: wilton_default
allowed_machine_colors: [5, 6, 8, 12]
reed_per_m: null              # warp ends per meter; loom constant. null -> square assumption + grid_unverified
density_definition: knots_m2  # knots_m2 | wefts_per_m | reed_x_wefts
allow_api: false              # true permits API backends (muse_image); customer photo leaves the machine
muse_provider: meta           # meta | fal | runware | openrouter
muse_price_usd: 0.01          # per generated image; used for cost_usd reporting only
photo_long_px_rec: 3000       # recommended to customers; informational only, never a gate
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
deltaE_warn: 3.0              # production: per-entry quantization_deltaE above this -> palette_far warning
eval_deltaE_max: 3.0          # offline eval: knot_deltaE_mean ceiling vs truth; fixed after designer measurement
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
photo_px_warn: 1.5            # applied to photo_px_per_knot_eff; the only resolution gate
photo_px_stop: 0.8
pile_blur_knots: 2.0
grid_mismatch_tol: 0.05
edge_density_min: 0.02
gradient_class_frac: 0.25
noise_class_frac: 0.08
class_margin_min: 0.2         # class_margin below this -> class_uncertain (auto mode) / no class_conflict (designer mode)
design_mode: quantize         # segment | quantize | edit; class files switch to segment/edit only after eval
edit_agent: null              # required when design_mode is edit; config/agents/<name>.yaml
edit_seed: 0
edit_px_per_knot_target: 2.0  # single-pass target; tiling kicks in below this when edit_tiling is auto
edit_px_per_knot_min: 1.0     # below this -> edit_low_res hard gate
edit_tiling: auto             # auto | off; auto tiles at edit_tile_px when single pass < target
edit_tile_px: 1024
edit_thin_line_max: 0.01      # thin_line_frac at or above this -> job is routed away from edit (edit_thin_lines)
edit_max_wall_s: 900          # soft cap for the agent loop; exceeded -> edit_timeout (best candidate so far is used)
edit_align_max: 6.0           # px at 4 px/knot (= 1.5 knots), p95 of block displacement; above -> edit_drift
edit_added_max: 0.01          # added-component knot share vs quantize baseline; above -> edit_added hard gate
edit_missing_max: 0.02        # missing-component knot share vs quantize baseline; above -> edit_missing hard gate
edit_gate_source: assumed     # assumed | eval; set to eval by eval/run.py once correlation with cc_* passes
edit_gate_corr_min: 0.6       # Spearman(edit_*_pct, cc_*) required before edit gates count as eval-sourced
edit_gain_tol: 0.02           # selected candidate must reach baseline_edge_iou - this, else edit_no_gain
edit_palette_deltaE_max: 8.0  # loop stop criterion; candidate colors must stay near the palette
edit_added_motif_max: 0       # offline eval: allowed added motifs for edit mode; 0 until designer measurement says otherwise
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

Başlangıç durumu (eğitim öncesi):
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
`classic.yaml` ve `geometric.yaml` boştur. `design_mode: segment` bir sınıf dosyasına ancak `segment` o sınıfta yayın kriterini tutturunca yazılır (commit, eval CSV referansı); `design_mode: edit` + `edit_agent: <ad>` agent yarışı kazananı için aynı kuralla. Bu sıralama sayesinde checkpoint gelmeden hiçbir iş `segment_fallback` uyarısı üretmez ve `design_mode_class_default` her zaman gerçekten çalışabilen bir modu gösterir.

## İplik kütüphanesi (opsiyonel)

```
yarn_code,name,R,G,B,material
12,Beige,204,183,152,Wool
```

---

## Veri (data/)

```
data/
├── photo_survey.csv          # existing customer photos: long_px, width_cm, height_cm, reed, px_per_knot_geo
├── real/
│   ├── pairs/<id>/{photo.jpg, truth.tiff, corners.json, H_refined.json, H_local.json,
│   │              align_mask.npy, gradient_mask.npy, noise_mask.npy, meta.json}
│   └── splits.json
├── archive/designs/<id>/{design.tiff, palette.csv}
├── backgrounds/*.jpg
└── synth/
    ├── dist.yaml              # real-world distributions (colors, size, density, px/knot); defaults until real pairs exist
    ├── pairs/<id>/{photo.jpg, truth.tiff, flat.png, meta.json, corners_gt.json, layout_gt.json}
    ├── generate.py            # design -> photo (render chain)
    └── generate_designs.py    # procedural designs with freely combined class features
```

`meta.json` (gerçek çift): `consent_train`, `reviewed`, `design_class` (desenci), `width_cm`, `height_cm`, `reed_per_m`, `wefts_per_m`, `machine_colors`, `photo_native_px`, `px_per_knot_geo`, `design_mode`, `edit_agent`, `qc` özellikleri, `qc_status`, `designer_score`, `fix_minutes`, `fix_type`.

### Fotoğraf taraması (`tools/photo_survey.py`)

Kod yazılmadan önce, eldeki ≥ 30 gerçek müşteri fotoğrafı ve ebatlarıyla: uzun kenar px, `px_per_knot_geo = long_px / max(W, H)` dağılımı (medyan, p10, `≥ photo_px_warn` oranı, `≥ photo_px_stop` oranı). Karar tablosu:
- `px_per_knot_geo ≥ 1.5` oranı ≥ %70 → müşteri fotoğrafları yeterli, çift toplama müşteri akışından.
- %30–70 → müşteri akışı + fabrika/desenci eldeki halıları yeniden çeker (telefon ≥ 12 MP, halıya dik, tam kadraj).
- < %30 → çift toplama tamamen yeniden fotoğraflamayla; müşteri akışında `low_photo_resolution` oranı kapasite raporunda izlenir ve müşteriye çekim kılavuzu gönderilir.
Sonuç `data/photo_survey.csv` ve `setup` bloğunda `photo_survey: bool`. Bu, kod yazımını bekleten tek dış girdidir; ölçüm bir günlük iştir.

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
- Çözünürlük: `pair_intake.py` `px_per_knot_geo` hesaplar; `< photo_px_stop` → ret; `< photo_px_warn` → kabul + `low_res_pair` etiketi (eğitimde çözünürlük augmentasyonunun gerçek örneği; `test`/`val`'e alınmaz). Uzun kenar piksel sayısı tek başına ret nedeni değildir.

Son üç madde en çok çift kaybettiren yerdir; `tools/pair_intake.py` bu alanları toplama anında doğrular, eksik olanı reddeder.

`train/align.py` ilk 20–30 çift gelince çalıştırılır, 200 çift sonra değil. Elenme oranı yüksekse toplama kriteri düzeltilir.

`tools/pair_stats.py` gerçek çiftlerden renk sayısı, ebat, tarak/atkı ve `photo_px_per_knot` dağılımlarını çıkarır → `data/synth/dist.yaml`. Sentetik üreteçler bu dosyadan örnekler; gerçek çift yokken `photo_survey.csv`'nin px/düğüm dağılımı ve varsayılan aralıklar (renk 2–12, ebat 60×90 – 300×400 cm, tarak 300–500) kullanılır, `dist.yaml.source: survey | default` yazılır.

### Bölünme

`test` ve `val` sadece gerçek; `train` = kalan gerçek + tüm sentetik. `splits.json` git'te kilitli, `design_class` ile katmanlı. Aynı halının farklı fotoğrafları aynı bölümde. `low_res_pair` etiketli çiftler sadece `train`'de. İki sabit çekirdek set:
- `test_core_classic`: ilk 20 `classic` test id.
- `test_core_modern`: ilk 20 modern test id. 20 gerçek modern çift birikene kadar 100 sabit prosedürel sentetik (`eval/synth_modern_ids.txt`); gerçek set dolunca ikisi de raporlanır.

### Gerçek çift hizalama (`train/prepare.py` → `train/align.py`)

Hizalama kodu `knotvision/align.py`'de tek fonksiyondur (`align_to_rectified(img, rectified) → warped, H_refined, H_local, align_px, align_mask`); `train/align.py` ve `design`/edit aynı fonksiyonu çağırır.
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
- `flat.png` = truth deseninin `rectified` uzayında (4W × 4H) düz render'ı (doku, gölge, perspektif yok). Edit LoRA hedefi; sentetikte `rectified` ile piksel-piksel hizalıdır çünkü perspektif zincirin sonrasında eklenir.

### Prosedürel desen üreteci (`data/synth/generate_designs.py`)

Klasik motif için arşiv gerekir; diğer dört sınıf arşivsiz üretilir. Sınıf özellikleri **serbestçe kombine edilir**: her desen bir taban şablondan başlar (`geometric | gradient | distressed | minimal`, kota ile) ve bağımsız olasılıklarla bordür, kafes tekrar, gradyan bölgesi (1–3), simplex gürültü (`noise_frac ∈ [0.05, 0.25]`), simetri ve ince çizgi (`thin_line_frac ∈ [0, 0.03]`) alır. Meta'ya `gradient_frac`, `noise_frac`, `thin_line_frac`, `has_border`, `has_repeat`, `symmetry`, `n_colors` yazılır.
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
   - **Atma birimi karo, çift değil.** `distressed` çiftler eğitime girer; `noise_mask` kapsaması > `tile_skip_noise_frac` (0.2) olan karolar atlanır, kalan karolar `loss_mask` ile kullanılır.
3. Augmentasyon: çevirme (çift birlikte), döndürme yok, renk jitter sadece girdi (±%10), çevrimiçi hav gölgelenmesi, lif bulanıklığı (0.3–3 düğüm), kamera gürültüsü, JPEG.
4. `train/data/{train,val}.jsonl` (`design_class`, karo listesi).
5. Hedefler (`edit` LoRA, opsiyonel) — **girdi ve hedef çıkarımla aynı uzayda**:
   - `edit_input.png` = çiftin `rectified.png`'si (perspektif düzeltilmiş, aydınlatma düzleştirilmiş, saçak kırpılmış), `design` adım 1 ile aynı kodla `edit_px_per_knot` çözünürlüğüne küçültülmüş ve 16 px katına yuvarlanmış. Ham fotoğraf **kullanılmaz**; çıkarımda model ham fotoğraf görmez.
   - `flat.png` = `truth.tiff` düz render'ı `rectified` uzayında (4W × 4H); gerçek çiftte `H_refined` ve `H_local` ile artık hizalama uygulanır, `align_mask == 0` blokları örnek ağırlığından düşülür; sonra aynı küçültme. Sentetikte `flat.png` zaten hizalıdır.
   - `layout_changed` çiftler dışarıda. Karo değil, tam görüntü (arka uç tam görüntü görür; `edit_tiling` çıkarımda kullanılacaksa eğitimde de aynı karo kesimi uygulanır, `tiled_train: true`).

### SegNet eğitimi (`train/train_segment.py`)
- `SegNet` (convnext_tiny features_only + FPN, stride 4, 4 kanal). Encoder ImageNet.
- Girdi 4 px/düğüm, 512×512 karo, 4 px hizalı, batch 16, AdamW 3e-4, cosine, 40 epoch, bf16. Karo örnekleme `design_class` dengeli.
- Kayıp: `loss_mask` maskeli L1(Lab) + λ·BCE(sınır), λ = 1.0.
- Augmentasyon: çevirme, renk jitter, ölçek 0.9–1.1 (ızgara hizalı), çevrimiçi aydınlatma/gölgelenme/bulanıklık/gürültü, çözünürlük augmentasyonu (1–4 px/düğüm'e küçült, 4'e büyüt; dağılım `photo_survey.csv`'den, müşteri fotoğraflarının gerçek px/düğüm aralığı ağırlıklı).
- Her 2 epoch'ta val: kayıp + sınıf başına 2 örnekte tam pipeline → eval metrikleri.
- Çıktı: `models/knotvision_segment.pt`.

### Edit LoRA eğitimi (`train/train_edit_lora.py`, opsiyonel)
- Sadece `qwen_edit_*` arka uçları için (`muse_image` eğitilemez). Amaç: taban prompt'la elde edilen sadakati artırmak; LoRA'sız agent yarışı taban çizgisidir, LoRA ancak yarışta ölçülebilir kazanç verirse agent dosyasına girer.
- `diffusers` + `peft`; MMDiT dikkat katmanlarına rank 32 LoRA; kodlayıcılar ve VAE donuk.
- Veri: sentetik (`edit_input.png` → `flat.png`) + hizalanan gerçek çiftler; prompt = taban prompt + sınıf eki + palet hex listesi (palet kartı görsel 2 olarak).
- Kiralık GPU (≥ 48 GB önerilir; 24 GB'da FP8 taban + gradient checkpointing). 3000–5000 adım, AdamW 1e-4, bf16, EMA yok.
- Her 500 adımda val: sınıf başına 2 örnekte tam pipeline (`qwen2511_direct` ile) → `edit_align_px`, `cc_added_*`, `knot_deltaE`.
- Çıktı: `models/edit_lora/<ad>.safetensors`; hash `versions.edit_lora_sha256`; agent dosyasında `edit_lora: <ad>`.

---

## Değerlendirme (eval/)

### Offline (`eval/run.py`)

`test_core_classic` + `test_core_modern` (+ `test_full`) + 100 sentetik (arşiv) + 100 sentetik (prosedürel), her mod (edit'te her agent), `regularize` açık/kapalı. **Sonuç tablosu `design_class` kırılımıyla**; toplam ortalama ek sütun. `muse_image` içeren çalışmalar API maliyeti yüzünden sadece `test_core_*` + 30 sentetik ile koşulur, tam set sadece açık ağırlık arka uçlarında.

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
| `class_acc` | `pure` sentetik ve etiketli gerçek: tespit doğruluğu, karışıklık matrisi; `class_uncertain` oranı; `class_margin` ile doğruluk ilişkisi (marj eşiği buradan doğrulanır) | sınıf tespiti |
| `class_mixed_recall` | `mixed` sentetik: `class_mixed` tetiklenme oranı; `pure`'da yanlış tetiklenme oranı | karışık halı uyarısı |
| `class_conflict_rate` | gerçek: desenci sınıfı ile otomatik sınıf uyuşmazlığı, sınıf kırılımlı | tespit kalitesi |
| `thin_line_frac_err` | sentetik: ölçülen `thin_line_frac` − meta | edit yönlendirme girdisi |
| `knot_deltaE_mean`, `_p95` | düğüm başına çıktı − truth ΔE2000; gradyan maskesinde 8×8 blok | renk |
| `banding` | gradyan maskesinde düşük frekanslı L farkı RMS | bantlaşma |
| `dither_err` | 4×4 blok karışım oranı farkı | dither |
| `noise_fidelity` | truth `noise_mask` içinde: (a) düğüm bazlı indeks uyuşma oranı `noise_index_acc`, (b) çıktı / truth küçük bileşen sayısı oranı `noise_cc_ratio`. `noise_fidelity = noise_index_acc × min(r, 1/r)`. Maske dışında `noise_1knot_frac(çıktı) − noise_1knot_frac(truth)`. | gürültü sadakati; `distressed`'ın tek gerçek kalite ölçüsü |
| `thin_line_recall` | truth `is_thin` bileşenlerinin korunma oranı; mod başına, o modun çalıştığı işlerde | ince çizgi |
| `cc_added_area_pct` | eşleşmeyen çıktı bileşenlerinin düğüm oranı (gradyan ve gürültü maskesi dışı) | uydurma (alan) |
| `cc_added_motif` | eşleşmeyen, alanı > `motif_min_knots`, `is_thin` olmayan çıktı bileşeni sayısı | uydurma motif (sert, 0; edit'te `edit_added_motif_max`) |
| `cc_missing_pct` | eşleşmeyen truth bileşeni oranı; aşama kırılımı `guard.json`'dan | eksik motif |
| `edge_iou` | 1 düğüm toleransla sınır IoU | konum |
| `line_jaggedness_delta` | çıktı − truth | düzenlilik |
| `free_len_share` | serbest parça uzunluk payı | vektörleştirme yeterliliği |
| `repeat_mismatch`, `symmetry_diff` | | tekrar, simetri |
| `regularize_changed_pct`, `guard_reverted_pct` | | yük |
| `edit_align_px` (medyan, p95) | edit adayının fotoğrafa ECC yer değiştirmesi | geometri kayması |
| `edit_gate_corr` | Spearman(`edit_added_pct`, `cc_added_area_pct`) ve Spearman(`edit_missing_pct`, `cc_missing_pct`); ≥ `edit_gate_corr_min` ise `edit_added_max` / `edit_missing_max` 95. yüzdelik kuralıyla profile yazılır, `edit_gate_source: eval` | referanssız kapıların geçerliliği |
| `edit_route_out_rate`, `edit_no_gain_rate` | `edit_thin_lines` yönlendirme oranı; taban çizgisini geçemeyen iş oranı | edit uygulanabilirliği |
| `edit_rounds`, `edit_candidate_spread` | agent döngü istatistikleri | agent verimi, kararsızlık |
| `cost_usd` | iş başına API maliyeti | maliyet |
| `qc_score`, `qc_status`, `qc_p5` | QC çıktısı; `knot_deltaE` ve desenci puanı ile korelasyon, sınıf başına | QC uyumu |
| `time_s` | toplam süre | |

Mod kararı sınıf başına verilir (sınıf dosyası `design_mode`); belirleyiciler `thin_line_recall`, `cc_added_*`, `cc_missing_pct`, `line_jaggedness_delta`, `banding`, `noise_fidelity`, `qc_score`; `edit` için ek olarak `edit_align_px`, `edit_route_out_rate`, `edit_no_gain_rate`, `time_s`, `cost_usd`.

### Agent yarışı (`eval/agents.py`)
`eval/run.py`'nin `design_mode: edit` için her `config/agents/*.yaml` ile koşulmuş halidir; `segment` ve `quantize` sonuçları aynı tabloya taban çizgisi olarak eklenir. Karolu/karosuz (`edit_tiling`) ayrı satırdır. Çıktı `eval/results/agents_<tarih>.csv` (sınıf × agent satırları, metrik sütunları) ve sınıf başına kazanan (kural: Kabul bölümü). Yarışın girdisi olan prompt şablonu, geri bildirim kuralları ve seçim ağırlıkları değişirse yarış tekrarlanır; yarış sonucu olmadan hiçbir agent `calibrated_configs`'a giremez.

### QC kalibrasyonu (`eval/calibrate_qc.py`)
Girdi `eval/score_sheet.csv` (`job_id, design_class, mode, edit_agent, regularize, designer_score, designer_score_2, fix_minutes, fix_type, notes`) + `qc.json`. Çıktı `models/qc_calib.json`:
```json
{"features": ["rerender_edge_iou", "palette_deltaE_p95", "rerender_color_deltaE_p95", "line_jaggedness",
              "noise_1knot_pct", "postprocess_changed_pct", "structure_mismatch", "banding"],
 "dropped_features": {"edge_width_knots": "n_cap", "illum_residual": "n_cap"},
 "class_levels": ["classic", "geometric", "gradient", "other"],
 "mode_levels": ["segment", "quantize", "edit"],
 "calibrated_configs": [["classic", "segment", null, true], ["geometric", "segment", null, true],
                        ["gradient", "segment", null, true], ["distressed", "quantize", null, false], ["minimal", "segment", null, true],
                        ["geometric", "edit", "qwen2511_loop", true], ["gradient", "edit", "qwen2511_loop", true]],
 "agent_hashes": {"qwen2511_loop": "..."},
 "model": {"type": "logistic_l2", "C": 0.3, "mean": [], "std": [], "coef": [], "intercept": 0.0},
 "model_p5": null,
 "target": 4, "qc_accept": 0.82, "qc_reject": 0.31,
 "cv_auc": 0.81, "cv_auc_by_class": {"classic": 0.84, "geometric": 0.83, "gradient": 0.70, "other": 0.76},
 "cv_auc_by_config": {"geometric/edit/qwen2511_loop": 0.78, "gradient/edit/qwen2511_loop": 0.68},
 "class_unreliable": [], "config_unreliable": [], "intra_rater_kappa": 0.72,
 "n_samples": 130, "class_hist": {"classic": 40, "geometric": 40, "gradient": 28, "distressed": 12, "minimal": 10},
 "score_hist": {"1": 12, "2": 34, "3": 34, "4": 28, "5": 22},
 "fix_type_hist": {"geometry": 22, "color": 38, "regularity": 20, "motif": 16, "gradient": 10, "noise": 6, "sheen": 12},
 "model_hashes": {"segment": "...", "quantize": "...", "qwen_edit_2511": "...", "edit_lora": null}, "date": "..."}
```
`fix_type_hist`'te `sheen` payı yüksekse parlama/hayalet motif sorunu yatırım listesine girer. `edit` konfigürasyonlarında `fix_type: motif` payı `segment`'ten belirgin yüksekse (`> 2×`) `edit_added_motif_max` gevşetilmez ve o konfigürasyon kalibrasyondan çıkarılır.

---

## Süre ve maliyet ölçümü (bench/)

`bench/infer.py`: aşama başına süre, mod başına (edit'te agent başına: tek aday süresi, karo başına süre, tur başına süre, toplam), 24 GB ve 48 GB kart, üç halı boyutu → `bench/timings.json`. Süre bütçesi ve mod düşürme mekanizması yoktur: `segment` ve `quantize` 200×300 cm'de 1 dakikanın altındadır; `edit` dakikalar mertebesindedir (aday × tur × karo × adım) ve saha kabulü "dakikalar yeterli"dir. Tek sınır agent döngüsündeki `edit_max_wall_s` (yumuşak). `timings.json` rapor ve regresyon takibi içindir; `muse_image` için `cost_usd` da yazılır.

---

## Testler (tests/, pytest)

- `test_color.py`: sRGB→Lab, ΔE2000 referans çiftleri.
- `test_intake.py`: üç `density_definition` formülü; `grid_input_mismatch`; `reed_per_m: null` → kare varsayımı + `grid_unverified`; `grid_rounding`; `machine_colors ∉ allowed` hata; gerçek işte `design_class` eksik → hata, `--class auto` sadece batch/synthetic; `photo_long_px_below_rec` sadece uyarı, iş durmaz; parametre katmanı (agent katmanı dahil, sadece `edit_*` alanlarını ezer); `design_mode: edit` + `edit_agent` eksik → hata; `muse_image` + `allow_api: false` → `api_not_allowed`; `setup` bloğu (`segment_ckpt`, `photo_survey` dahil).
- `test_rectify.py`: sentetik köşe hatası < 8 px (`@pytest.mark.gpu`); `corners_override`; TPS bozulmuş sentetikte `non_planar`; saçak ± 3 px, kaba dörtgen `minAreaRect`'ten, şerit orijinal maskeye doğru geri taşınır; çıktı tam `4W × 4H`; aydınlatma düzleştirme; `rectified.png` CLAHE içermez; `edge_width_knots` ± 0.3; `pile_blur`; efektif çözünürlük kapıları; 1800 px uzun kenarlı sentetik `photo_px_per_knot_eff ≥ photo_px_stop` ise geçer.
- `test_structure.py`: `layout_gt` ile bordür/periyot/tekrar; köşe blokları `regions == 4`; panel desende bordür kabul edilmez; `coarse_labels.npy` W×H boyutunda ve `regions.npy` bileşen sınırları buradan (512 px maske ile fark ≤ 1 düğüm); gradyan maskesi IoU ≥ 0.8; sınıf tespiti `pure` prosedürel setlerde ≥ %90; `class_margin` her kuralda hesaplanır ve eşiğe yakın sentetikte `class_uncertain` tetiklenir; motifsiz düz degrade `gradient` olur, `pictorial` olmaz; gradyanlı+gürültülü sentetikte `distressed` seçilir ve `class_mixed` tetiklenir; `pile_blur` varken `distressed` seçilmez; kamera gürültülü temiz desende `noise_frac` eşik altı kalır, `noise_1knot_frac` yükselir; `class_conflict` desenci sınıfı farklıyken tetiklenir; `thin_line_frac` ± 0.005; `edit` istenen işte `thin_line_frac ≥ edit_thin_line_max` → `edit_thin_lines` ve mod sınıf varsayılanına döner; checkpoint yokken `segment_fallback` burada üretilir; `knot_grid.visible`, `grid_mismatch`, `measured_reed_per_m` ± %3; TPS sonrası `coarse_labels` yenilenir; `unmeasurable`.
- `test_palette.py`: komşu benzer kümeler ayrı, komşu olmayan benzerler birleşir; motif rengi korunur; gradyan uçları palette; `palette_used = 2` iki renkli desende; `palette_used ≤ machine_colors`; `gain_stop_select` ve `merge_cost_stop` ayrı yollarda ayrı ayrı durdurur; `regions.npy` girdi değildir; `palette_card.png` giriş sayısı kadar kare ve hex etiket içerir.
- `test_design.py`: `segment` çıkış girdi/4; karo birleştirme ΔLab ortalama < 0.5, p95 < 2.0; determinizm toleranslı; sentetik gradyan bölgede boundary < 0.3; çıkarım hatasında `segment_fallback` → quantize.
- `test_edit.py` (arka uç sahte/mock ile, GPU'suz): taban çizgisi her edit işinde önce üretilir ve agent değişince cache'ten gelir; `edit_px_per_knot` hesabı, `edit_tiling: auto` karo sayısı ve harmanlama, `edit_tiling: off` + düşük çözünürlük → `edit_low_res`; `edit_input.png` 16 px katı; çıktı tam `4W × 4H`; sahte arka uç bilinen homografiyle kaydırılmış görsel döndürünce hizalama geri bulur ve `align_px` ± 1 px; kaydırma `edit_align_max` üstündeyse `edit_drift`; sahte arka uç fotoğrafta olmayan blok eklerse `edit_added_pct` yükselir (referans `ref_indexed`) ve geri bildirim prompt'una "Remove any element..." cümlesi girer; motif silerse `edit_missing_pct` yükselir ve `qc`'de `edit_missing` sert kapısı düşer; hav dokulu sentetikte `ref_indexed` tabanlı `edit_added_pct`, `coarse_labels` tabanlı hesaptan düşük çıkar (referans gürültüsü testi); eşikleri geçen aday ilk turda döngüyü durdurur; `edit_max_rounds` aşılmaz; `edit_max_wall_s` aşımı → `edit_timeout` + en iyi aday; sahte arka uç taban çizgisinden kötü döndürünce `edit_no_gain` ve çıktı = taban çizgisi; eşit skorda en düşük `(r, k)`; `qwen_edit_*` seed formülü karo indeksi dahil; arka uç üç kez hata verince `edit_fallback` → taban çizgisi, ek hesap yok; `muse_image` `deterministic: false`, `api_egress: true`, `cost_usd = n × karo × fiyat`; prompt sınıf ekleri doğru sınıfta girer; `stages/05_design/{baseline,edit}/` dosyaları yazılır. `@pytest.mark.gpu`: `qwen_edit_2511` gerçek çıkarım, aynı seed iki çalıştırma ΔLab p95 < 2.0.
- `test_agents.py`: her `config/agents/*.yaml` şema doğrulaması (bilinen `backend`, `select_weights` anahtarları özellik setinde, `edit_n_candidates ≥ 1`); agent hash prompt şablonu değişince değişir; `eval/agents.py` sınıf başına kazanan kuralı (kriteri tutturan yoksa `null`; eşitlikte ucuz); en fazla iki agent `calibrated_configs`'a girer; `edit_gate_source: assumed` konfigürasyon kalibrasyona alınmaz.
- `test_production.py`: 3 renk kuantizasyon; pooling eşitlik kuralı; 1×1 ve 1×2 ada silinir, 1×10 şerit korunur; `is_thin` tek fonksiyon; iki geçiş üçüncü giriş oluşturmaz; kapsama birleştirmesi motif rengini silmez; gradyan maskesinde temizleme atlanır; `min_region_knots: 0` atlar; `min_region_knots: 1` sadece tek düğümlük adaları siler; `noise_1knot_pct` doğru; `edit` modunda `edit_added_pct`/`edit_missing_pct` `ref_indexed`'e göre hesaplanır; `palette_far` uyarısı; `stages/guard.json` yazılır.
- `test_guard.py`: alanı > `motif_min_knots` bileşen kaybolunca sadece o bileşen geri yazılır; ≤ eşik geri yazılmaz; 11–20 düğüm bandı geri yazılır; gradyan maskesi ve distressed'ta no-op; `guard_heavy`; aşama kırılımı.
- `test_regularize.py`: tırtıklı 1:1 diagonal → temiz basamak; ±1 gürültülü düz çizgi → düz; yay artık ≤ tolerans; serbest parça run-length monoton parçalı sabit, uçlar sabit; tekrar periyotlarında tek düğüm hatası silinir; köşe bloğu dokunulmaz; `repeat_mismatch = 0` ve şablon bir kez düzleştirilir; kapsanmayan düğüm `indexed_raw` indeksini korur; `is_thin` iskelet korunur; eşitlikte en düşük indeks; `regularize: off` istatistik üretir; truth desenlerde `changed_pct ≈ 0`; `segment_classes` toplamı 1.
- `test_dither.py`: sentetik rampa → 4×4 blok karışım oranı ± 1/levels; deterministik; maske dışına taşmaz; sınır şeridi uç indekse; `dither_pattern: null` → uyarı; `dither_stats.py` düzeni doğru çıkarır.
- `test_qc.py`: truth'tan üretilen desende `rerender_edge_iou` yüksek; `qc_calib.json` yokken `needs_review`; sert kapılar (`class_mixed`, `mode_uncalibrated`, `edit_drift`, `edit_added`, `edit_missing`, `edit_low_res`, `edit_timeout`, `edit_fallback`, `edit_no_gain` dahil); `mode_uncalibrated` agent hash değişince tetiklenir; çekirdek özellik birleştirmeleri (`postprocess_changed_pct`, `structure_mismatch`) ve null → standardize 0; `status` `@target`; `reasons` 3 katkı; hash uyuşmazlığı; `class_unreliable`, `config_unreliable`.
- `test_calibrate.py`: L2 lojistik model ve `C` seçimi; `qc_target` seçimi; `p5` sadece yeterli örnekle; `n_özellik ≤ n/15` kuralı öncelik sırasıyla düşürür; `class_levels`, `mode_levels`; > %50 null özellik düşer; kappa; sınıf ve konfigürasyon başına AUC; `calibrated_configs` dörtlü ve `agent_hashes` yazılır; kapasite medyan/p90 mod kırılımlı; `cost_usd` medyanı.
- `test_align.py`: bilinen homografi ECC ile geri bulunur; yerel bozulmada `align_mask`; truth dither → `gradient_mask`; truth gürültü → `noise_mask`; `knotvision/align.py` ve `train/align.py` aynı sonucu verir.
- `test_prepare.py`: `distressed` çift eğitime girer; `noise_mask` kapsaması > 0.2 karolar atlanır, kalanlar `loss_mask` ile; `loss_mask = align ∧ ¬noise`; edit LoRA girdisi `rectified`'dan üretilir (ham fotoğraftan değil) ve `design` adım 1 ile aynı boyutta; `flat.png` `H_local` ile hizalı; `layout_changed` dışarıda.
- `test_metrics.py`: `cc_added_*`, `cc_missing_pct`; kuantizasyon artığı (3–10 düğüm) artırmaz; `thin_line_recall`; `line_jaggedness`; `repeat_mismatch`; `banding` bantlı > dither'lı; `noise_fidelity`: gürültü silinmiş çıktıda düşük, birebir çıktıda 1, fotoğraf gürültüsü eklenmiş çıktıda `noise_cc_ratio > 1`; `edge_fidelity` tek fonksiyon; `edit_gate_corr` hesabı ve eşik türetme.
- `test_tools.py`: `motif_stats`, `dither_stats`, `noise_stats` (sentetik eskitme deseninde `noise_cc_min/max` ve tek düğüm payını doğru çıkarır), `pair_stats` (`dist.yaml`), `pair_intake` (eksik alan reddi; `px_per_knot_geo < photo_px_stop` reddi; 1800 px uzun kenarlı yeterli çift kabul edilir ve gerekirse `low_res_pair` etiketi alır), `photo_survey` (karar tablosu eşikleri), `setup-check` çıktısı (edit arka uçları, `segment_ckpt`, `photo_survey` dahil).
- `test_generate.py`: prosedürel `pure` ve `mixed` örnekler meta ile tutarlı; `thin_line_frac` meta ile ± 0.005; `dist.yaml` örnekleme aralıkları; `flat.png` truth ile piksel bazında tutarlı ve `rectified` ile hizalı.
- `test_export.py`: `design.tiff` yazılıp okunur; anizotropik `resolution`; indeks sayısı ≤ `machine_colors`; `palette.csv` sütunları.
- `test_cache.py`: aynı girdi ikinci `run` hiçbir aşamayı çalıştırmaz; `rerun --from production --palette` sadece production ve sonrası; sınıf dosyası değişince `structure` sonrası yeniden çalışır; agent dosyası değişince sadece `design/edit` ve sonrası yeniden çalışır, `design/baseline` cache'ten gelir; `edit_seed` cache anahtarında; `muse_image` adayları cache'ten `rerun --from production` ile API çağrılmadan kullanılır.

---

## Kod yapısı

```
knotvision/
├── config/
│   ├── looms/wilton_default.yaml
│   ├── classes/{classic,geometric,gradient,distressed,minimal}.yaml
│   ├── agents/{qwen2511_direct,qwen2511_loop,qwen2509_loop,muse_loop}.yaml
│   ├── dither/{ordered4.yaml, archive.yaml}
│   └── yarns/sample.csv (optional)
├── knotvision/
│   ├── job.py          Job dataclass, runs/, hash cache, retention, rerun, parameter layering (incl. agent layer), routing rules, setup block
│   ├── color.py        sRGB/Lab, deltaE2000
│   ├── intake.py       density definitions -> grid, validation, edit/agent/api validation
│   ├── rectify.py      SAM 3.1 wrapper, manual corners, coarse quad + fringe, planarity, homography, white balance, illumination, edge sharpness
│   ├── structure.py    layout, border acceptance, gradient mask, coarse labels at grid resolution, class detection + margin, thin_line_frac, knot grid field, TPS, regions
│   ├── palette.py      photo palette (adjacency-aware merge, gradient endpoints, motif-color protection), yarn library matching, palette card
│   ├── align.py        ECC global + local alignment to rectified; shared by design_edit and train/align.py
│   ├── design_seg.py   SegNet (convnext_tiny + FPN, stride 4), grid-aligned tiling
│   ├── design_quant.py 4 px/knot mean-shift; also the edit baseline
│   ├── design_edit.py  edit mode entry: baseline + ref_indexed, resolution, tiling, calls agents/loop, gain check, writes design_rgb + edit report block
│   ├── agents/
│   │   ├── base.py     Backend protocol: edit(images, prompt, negative_prompt, seed, params) -> PIL image; max_long_px; supports_seed
│   │   ├── loop.py     candidate generation, align, quick quantize, features vs ref_indexed, weighted selection, feedback, stop/timeout, cost
│   │   ├── prompts.py  base prompt, class additions, feedback rules
│   │   └── backends/{qwen_edit.py (2511/2509, diffusers, optional LoRA, Lightning), muse_image.py (Meta Model API + fal/Runware/OpenRouter adapters)}
│   ├── production.py   two-pass quantization, mode pooling, noise measurement, edit added/missing vs ref_indexed, cleanup, coverage merge
│   ├── guard.py        assert_no_motif_loss, stages/guard.json
│   ├── regularize.py   template-first repeat/symmetry merge, vectorization, angle snap, arc fit, free-run regularization, re-rasterization
│   ├── dither.py       analytic ordered dither inside the gradient mask
│   ├── metrics.py      is_thin, edge_fidelity, line_jaggedness, repeat_mismatch, banding, noise_fidelity, cc_*, thin_line_recall, edit_added/missing
│   ├── qc.py           hard gates (incl. edit_*), rerender, raw measurements, core feature merge, L2 logistic score, reasons
│   ├── export.py       design.tiff + palette.csv
│   └── run.py
├── tools/ photo_survey.py, motif_stats.py, dither_stats.py, noise_stats.py, pair_stats.py, pair_intake.py
├── data/ (photo_survey.csv, real, archive, backgrounds, synth)
├── train/ prepare.py, align.py, train_segment.py, train_edit_lora.py, align_report.csv, runs/
├── eval/ run.py, agents.py, calibrate_qc.py, capacity.py, score_sheet.csv, synth_test_ids.txt, synth_modern_ids.txt, results/
├── bench/ infer.py, timings.json
├── tests/
├── models/ knotvision_segment.pt, qc_calib.json, sam3.1/, qwen_image_edit_2511/, edit_lora/ (gitignore)
├── cli.py   run | rerun --from <stage> [--class c] [--palette ...] [--confirm-palette] [--mode segment|quantize|edit]
│            [--agent <name>] [--seed n] [--corners ...] [--no-gate] [--regularize on|off] [--dither on|off]
│            | setup-check | batch --class auto [--mode ...] [--agent ...]
├── requirements.txt
└── CLAUDE.md
```

## Bağımlılıklar

```
numpy, opencv-contrib-python, pillow, scikit-image, scipy, scikit-learn, tifffile, pyyaml
torch, torchvision, timm
diffusers, transformers, accelerate     # qwen_edit backends (QwenImageEditPlusPipeline)
peft                                    # optional, edit LoRA training/loading
requests                                # muse_image API adapters
segmentation-models-pytorch             # optional, fallback decoder only
sam3   # pip install git+https://github.com/facebookresearch/sam3 ; weights in models/sam3.1/ (access granted)
noise
pytest
```

## Uygulama sırası

Kod yazımını bekleten tek dış girdi fotoğraf taramasıdır (adım 0, bir gün). Diğer kurulum verisi geldikçe varsayılanların yerine geçer.

0. **Fotoğraf taraması:** `tools/photo_survey.py` eldeki ≥ 30 müşteri fotoğrafı + ebat üzerinde → `px_per_knot_geo` dağılımı → çift toplama kaynağı kararı (müşteri akışı / yeniden fotoğraflama / karışık). Bu araç pipeline'dan bağımsızdır (EXIF + aritmetik), ilk gün yazılır.
1. **Kurulum ve arşiv araçları:** tezgah profili; varsa iplik CSV; SAM 3.1 checkpoint `models/sam3.1/`. `motif_stats.py`, `dither_stats.py`, `noise_stats.py` arşiv üzerinde → `motif_min_knots`, `config/dither/archive.yaml`, `noise_cc_min/max`, distressed `min_region_knots`; desenciye teyit. `setup-check` ile eksikler listelenir.
2. `generate_designs.py` (prosedürel, karışık özellikli, ince çizgi dahil) — arşive bağımlı değil; `test_core_modern` sentetik sürümü buradan. `dist.yaml` tarama dağılımıyla. `flat.png` çıktısı dahil.
3. Modelsiz uçtan uca hat: `intake → rectify (SAM 3.1) → structure → palette → quantize → production → regularize → dither → qc (özellikler, karar yok) → export`. Profil varsayılanı `quantize` olduğu için hiçbir iş `segment_fallback` üretmez. Texcelle'de açılır. `reed_per_m: null` ise ilk işlerin `measured_reed_per_m` değeriyle profil doldurulur.
4. **Çift toplama başlar** (`pair_intake.py` ile, sınıf kotalı, kaynak adım 0 kararına göre). İlk 20–30 çiftte `align.py` → `align_report.csv`; elenme oranı yüksekse kriter düzeltilir. `pair_stats.py` → `dist.yaml`.
5. **Desenci ölçümü (model eğitiminden ve agent yarışından önce):** 10 gerçek iş — 3 `classic`, 3 `geometric`, 2 `gradient`, 1 `distressed`, 1 `minimal` — modelsiz hattan, `regularize` açık/kapalı iki varyant. Süre, puan, düzeltme türü (`sheen` ve `noise` dahil). Bu ölçüm: projenin işe yarayıp yaramadığını, yatırımın nereye gideceğini, kapasite formülünün ilk girdisini, `jag_max`, `banding_max`, `eval_deltaE_max` başlangıçlarını verir. Sonuç gelmeden 8. ve 9. adıma geçilmez.
6. `bench/infer.py` → `timings.json`.
7. `generate.py` + offline eval, sınıf kırılımlı. Modelsiz hat = taban çizgisi.
8. **Edit arka uçları ve agent yarışı:** `qwen_edit_2511` yerel kurulum (kiralık GPU), `muse_image` API (fabrika `allow_api` kararı); `qwen2511_direct` ile 5 gerçek iş üzerinde duman testi (Texcelle'de açılıyor mu, `edit_align_px` ölçülebiliyor mu, `edit_no_gain` oranı). Ardından `eval/agents.py`: hazır agent'lar × sınıf × karolu/karosuz, `test_core_*` + sentetik; `muse_image` kısıtlı set. `edit_gate_corr` ölçülür; eşik sağlanırsa `edit_added_max` / `edit_missing_max` profile yazılır. Sınıf başına kazanan belirlenir; kazanan taban çizgisini geçmiyorsa o sınıfta `edit` kullanılmaz. İsteğe bağlı: `train_edit_lora.py` ile LoRA, yarış tekrarı.
9. `segment` eğitimi (prosedürel set + hizalanan gerçek çiftler, karo bazlı atlama). Eval; yayın kriterine iki çekirdek sette bak. Kriteri tutturan sınıfların dosyalarına `design_mode: segment` yazılır. `distressed`'ta `noise_fidelity` quantize'ı geçerse sınıf dosyası güncellenir. `free_len_share` klasikte > 0.6 ise öğrenilmiş vektörleştirme gündeme alınır. Adım 8 ve 9 paralel yürütülebilir; aynı `test_core` setlerinde karşılaştırılır.
10. **Mod kararı sınıf başına:** `segment` ↔ kazanan `edit` agent ↔ `quantize`; sınıf dosyalarına yazılır.
11. QC kalibrasyonu: sınıf kotalı ≥ 100 çıktı + kazanan `edit` konfigürasyonları için sınıf başına ≥ 10 (en fazla iki agent, `edit_gate_source: eval` şart), intra-rater kontrolü → `qc_calib.json`. Bundan sonra pipeline karar verir.

## Tasarım kararları

- **Taban çizgisi her yerde.** `quantize` profil varsayılanıdır, `edit`'in referansı, yedeği ve kıyas noktasıdır, `segment`'in hata yedeğidir. Sınıf dosyalarına model gerektiren mod ancak o model o sınıfta ölçülmüş kazanç verince yazılır; varsayılan hiçbir zaman var olmayan bir checkpoint'e işaret etmez.
- **Referanssız edit kapıları taban çizgisine karşı ölçülür, ham kuantizasyona karşı değil.** Aday ve referans aynı palet ve aynı kuantizasyon yolundan geçer; hav dokusu, gölge ve parlama bileşenleri iki tarafta da aynı şekilde elendiği için fark gerçek ekleme/silmeyi yansıtır. Eşikler `cc_*` ile korelasyon ölçülmeden `assumed` kalır ve `assumed` eşikli konfigürasyon kalibre edilemez.
- **Motif silinmesi edit'te de sert kapıdır.** `edit_missing` `edit_added` ile simetriktir; guard indeks haritası olmayan `design` aşamasında çalışamadığı için kapı `production` adım 3'teki nihai değerle kurulur.
- **İnce çizgi edit'e gitmez.** Diffusion arka ucu ≤ 2 px/düğüm'de 1 düğümlük çizgiyi taşıyamaz; bunu kriter gevşeterek değil yönlendirmeyle (`thin_line_frac`) çözmek daha doğrudur. Karolama otomatiktir ve büyük halıda 2 px/düğüm'ü garanti eder.
- **Agent'ın eleştirmeni ölçümdür, dil modeli değil.** Geri bildirim prompt'u sayılardan kurallarla üretilir; agent davranışı test edilebilir ve determinist kalır (arka uç hariç).
- **Tek aday seçim ölçütü: ağırlıklı skor.** Kalibre QC modeli iş düzeyi özelliklerle fit edilir; aday özellik setiyle uyumsuz olduğu için aday seçiminde kullanılmaz.
- **QC modeli küçük ve sabit.** 130 örnekle ~20 özellikli ordinal model aşırı uyum verir; çekirdek set ≤ 10, `n_özellik ≤ n/15` kuralı, L2 lojistik regresyon. `edit_*` büyüklükleri kapıdır, modele girmez.
- **Sınıf tespiti marjla raporlanır.** Kuralların skoru yoktur; her kuralın eşiğe normalize marjı vardır. `class_uncertain` ve `class_conflict` bu marja bağlıdır.
- **Çözünürlük kapısı tek: efektif px/düğüm.** Uzun kenar piksel eşiği müşteri fotoğraflarının çoğunu ve çift toplamayı elerdi; fotoğraf taraması bu kararı veriye bağlar.
- **Eğitim ve çıkarım aynı uzayda.** Edit LoRA girdisi `rectified`'dır, hedef `rectified` uzayında hizalanmış `flat.png`'dir.
- **API arka ucu için üç şart:** `allow_api` profil kararı, `cost_usd` raporu, `api_egress` raporu.
- **`cc_added_motif` edit'te ayarlanabilir, diğerlerinde 0.** "Benzer yeterli" kararı yalnızca diffusion çıktısı için anlamlıdır; gevşetme desenci ölçümüne bağlıdır.
- **Süre bütçesi ve mod düşürme yok.** `edit` dakikalar sürer, saha bunu kabul eder; tek koruma `edit_max_wall_s` yumuşak sınırı ve `edit_no_gain` kontrolü.
- **Tek sınıf etiketi.** Karışık halıyı sınıf değil, sınıftan bağımsız gradyan maskesi ve `class_mixed` uyarısı taşır.
- **Sınıf desenciden.** Gerçek işte otomatik tespit karar vermez, rapor eder ve çelişkiyi işaretler.
- **Gradyan maskesi muafiyetleri sınıf dosyasında değil, aşama kodunda.**
- **İşlem iplikleri pipeline dışı.** Palet makinenin renk sınırının tamamını kullanır.
- **`regions.npy` tek tüketicili (`regularize`).**
- **`gain_stop` ikiye ayrıldı; `deltaE_warn` ikiye ayrıldı.** Kütüphaneli/kütüphanesiz palet yolları ve üretim uyarısı/eval kriteri farklı büyüklükleri ölçer, aynı sayıyı paylaşamaz.
- **Açık nokta yerine kurulum.** Varsayılanla çalışan her alan sert kapı üretir, sessizce geçmez; `setup-check` listeler.
- **Parlama / hayalet motif için mekanizma yok.** Ölçüm (`fix_type: sheen`) var, kod yok.
- **Hizalama kodu tek yerde (`knotvision/align.py`); `is_thin` tek yerde (`metrics.py`).**

## Sonraki sürüm adayları

VLM eleştirmenli agent (Qwen3-VL, Apache 2.0: adayı fotoğrafla karşılaştırıp bölge bazlı geri bildirim yazan eleştirmen; `critic: vlm`), Qwen-Image-2.0 arka ucu (ağırlık yayınlanırsa), karolu edit için dikiş düzeltme, edit'te taban çizgisinden bileşen geri yazma (guard'ın `design` aşamasına uzantısı; `edit_missing` kapısı sistematik düşerse), aday düzeyi kalibre seçim modeli, retrieval (DINOv2), rulo tekrar, `pictorial` sınıfı, diğer tezgah profilleri, sınır düğümleri için sınıflandırma başlığı, palet koşullu SegNet başlığı, QC için öğrenilmiş geri-render, öğrenilmiş vektörleştirme (`free_len_share` > 0.6 ise), gradyan bölgesinde öğrenilmiş rampa kestirimi, parlama/hayalet motif için desenci girdili motif işaretleme (`sheen` payı yüksekse).
