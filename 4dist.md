Berikut ringkasan parameter untuk keempat distribusi tersebut beserta library Python yang umum digunakan.
Distribusi Normal membutuhkan dua parameter: rata-rata (μ\mu
μ) yang menentukan posisi pusat kurva, dan deviasi standar (σ\sigma
σ) yang menentukan lebar sebaran. Distribusi ini simetris dan biasa digunakan untuk memodelkan kekuatan material atau variabel yang sebarannya merata di kedua sisi rata-rata.
Distribusi Lognormal juga menggunakan dua parameter, namun keduanya didefinisikan dalam skala logaritma dari data: μ\mu
μ adalah rata-rata logaritma data dan σ\sigma
σ adalah deviasi standar logaritma data dengan rumus PDF f(x)=1(2π)1/2σxexp⁡[−(ln⁡x−μ)22σ2]f(x) = \frac{1}{(2\pi)^{1/2}\sigma x} \exp[-\frac{(\ln x - \mu)^2}{2\sigma^2}]
f(x)=(2π)1/2σx1​exp[−2σ2(lnx−μ)2​] untuk x≥0x \geq 0
x≥0. Distribusi ini cocok untuk data yang selalu positif dan condong ke kanan (skewed), seperti waktu hingga kegagalan atau intensitas korosi. ResearchGate
Distribusi Weibull membutuhkan dua hingga tiga parameter: parameter bentuk (shape, cc
c atau kk
k), parameter skala (scale), dan opsional parameter lokasi (loc) untuk menggeser distribusi di mana weibull_min menggunakan c sebagai parameter bentuk (disebut k pada referensi Wikipedia). Yang menarik, Weibull dengan c=1 akan tereduksi menjadi distribusi eksponensial, dan c=2 akan tereduksi menjadi distribusi Rayleigh — sehingga Rayleigh sebenarnya adalah kasus khusus dari Weibull. Distribusi ini sangat fleksibel dan banyak dipakai dalam reliability engineering karena bentuknya bisa disesuaikan untuk berbagai pola kegagalan. ResearchGateResearchGate
Distribusi Rayleigh hanya membutuhkan satu parameter skala (σ\sigma
σ), karena bentuknya sudah tetap. Dalam konteks kelautan, distribusi ini paling umum dipakai untuk memodelkan tinggi gelombang individual dalam satu kondisi laut (short-term sea state), berbeda dengan Weibull yang lebih cocok untuk distribusi jangka panjang (long-term) tinggi gelombang signifikan.
Untuk library Python, scipy.stats adalah pustaka standar yang dipakai hampir di semua kasus di mana scipy.stats mendefinisikan objek seperti lognorm, weibull_min, gamma, dan beta, dan setiap objek punya fungsi pdf(x, a, [b,] loc, scale) dan cdf(x, a, [b,] loc, scale). Untuk estimasi parameter dari data riil, scipy.stats menyediakan fungsi fit(data) yang menghitung parameter distribusi menggunakan metode maximum likelihood berdasarkan data empiris. ScienceDirectScienceDirect
Berikut tabel ringkasan:
DistribusiParameterFungsi scipy.statsKasus penggunaan umumNormalμ\mu
μ (mean), σ\sigma
σ (std)scipy.stats.normKekuatan material, error pengukuranLognormalμ\mu
μ, σ\sigma
σ (skala log)scipy.stats.lognorm (param s, loc, scale)Waktu kegagalan, variabel selalu positif & skewedWeibullshape (cc
c/kk
k), scale, locscipy.stats.weibull_minReliability engineering, kecepatan angin, fatigue lifeRayleighscale (σ\sigma
σ)scipy.stats.rayleighTinggi gelombang short-term, magnitudo vektor 2D acak