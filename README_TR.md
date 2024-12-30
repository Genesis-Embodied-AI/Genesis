![Genesis](imgs/big_text.png)

![Teaser](imgs/teaser.png)

[![PyPI - Version](https://img.shields.io/pypi/v/genesis-world)](https://pypi.org/project/genesis-world/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/genesis-world)](https://pypi.org/project/genesis-world/)
[![GitHub Issues](https://img.shields.io/github/issues/Genesis-Embodied-AI/Genesis)](https://github.com/Genesis-Embodied-AI/Genesis/issues)
[![GitHub Discussions](https://img.shields.io/github/discussions/Genesis-Embodied-AI/Genesis)](https://github.com/Genesis-Embodied-AI/Genesis/discussions)

[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README.md)
[![README en Français](https://img.shields.io/badge/Francais-d9d9d9)](./README_FR.md)
[![한국어 README](https://img.shields.io/badge/한국어-d9d9d9)](./README_KR.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README_CN.md)
[![日本語版 README](https://img.shields.io/badge/日本語-d9d9d9)](./README_JA.md)

# Genesis

## 🔥 Güncellemeler

- [2024-12-25] Işın izleme görüntüleyicisi desteğini de kapsayan bir [docker](#docker) eklendi.
- [2024-12-24] Genesis'e katkıda bulunmak için çeşitli [yönergeler](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/CONTRIBUTING.md) eklendi.

## İçerikler

1. [Genesis Nedir?](#genesis-nedir)
2. [Öne Çıkan Özellikler](#öne-çıkan-özellikler)
3. [Hızlı Kurulum](#hızlı-kurulum)
4. [Docker](#docker)
5. [Dokümentasyon](#dokümantasyon)
6. [Genesis'e Katkıda Bulunmak](#genesise-katkıda-bulunmak)
7. [Destek](#destek)
8. [Lisans ve Teşekkür](#lisans-ve-teşekkür)
9. [Alakalı Makaleler](#alakalı-makaleler)
10. [Alıntı](#alıntı)

## Genesis Nedir?

Genesis, genel amaçlı *Robotik/Bedenleşmiş Yapay Zeka/Fiziksel Yapay Zeka* uygulamaları için tasarlanmış bir fizik platformudur. Bu platform, aşağıdaki özelliklere sahiptir:

1. Çeşitli materyalleri ve fiziksel olayları simüle edebilen, sıfırdan inşa edilmiş bir **evrensel fizik motorudur**.
2. **Hafif**, **ultra-hızlı**, **kullanıcı dostu** ve **Pythonik** bir robotik simülasyon platformudur.
3. Güçlü ve hızlı bir **foto-gerçekçi işleme sistemidir**.
4. Kullanıcıdan gelen doğal dil tanımlamalarını çeşitli veri biçimlerine dönüştüren üretken bir **veri motorudur**. 

Sıfırdan tasarlanarak yeniden inşa edilen evrensel bir fizik motoru ile güçlendirilen Genesis, çeşitli fizik çözücülerini ve bu çözücülerin bağlantılarını birleşik bir framework'e entegre eder. Bu çekirdek fizik motoru, robotik ve ötesi için otomatik olarak veri üretimini amaçlayan, üst düzeyde çalışan bir generative agent framework ile daha da geliştirilmiştir.

**Not**: Şu anda _temel fizik motoru_ ve _simülasyon platformunu_ açık kaynak olarak kullanıyoruz. Geliştirdiğimiz _generative framework_ , her biri üst düzey bir agent tarafından yönlendirilen belirli bir dizi veri biçimini işleyen birçok farklı üretken modülü içeren modüler bir sistemdir. Modüllerin bazıları var olan makaleleri entegre ederken, bazıları da halen başvuru aşamasındadır. Üretken özelliğimize erişim yakında kademeli olarak sunulacaktır. İlgileniyorsanız, [paper list](#alakalı-makaleler) bölümünde daha fazlasını keşfedebilirsiniz.

Genesis'in hedefleri arasında;

- Fizik simülasyonlarını kullanmanın önündeki engeli azaltarak robotik alanındaki araştırmaları herkes için erişilebilir kılmak, ([Misyonumuzu inceleyin](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/mission.html)
- Fiziksel dünyayı en yüksek doğrulukla yeniden yaratmak için farklı fizik çözücülerini tek bir framework'de birleştirin.
- Veri üretimini otomatikleştirerek insan eforunu azaltın.

Proje sayfasına ulaşmak için: <https://genesis-embodied-ai.github.io/>

## Öne Çıkan Özellikler

- **Hız**: Tek bir RTX 4090 ile Franka robot kolunun simülasyonunda 43 milyondan fazla FPS elde edildi (gerçek zamandan 430.000 kat daha hızlı).
- **Çapraz Platform**: Linux, macOS, Windows üzerinde çalışır ve birden fazla compute backend (CPU, Nvidia/AMD GPU'lar, Apple Metal) desteği.
- **Farklı fizik çözücülerinin entegrasyonu**: Rijit gövde, MPM, SPH, FEM, PBD, Durgun Akışkan.
- **Geniş materyal çeşitliliği**: Katı cisimlerin, sıvıların, gazların, deforme olabilen cisimlerin, ince kabuklu cisimlerin ve tanecikli malzemelerin simülasyonu ve bağlanması.
- **Farklı robotlarla uyumluluk**: Robotik kollar, bacaklı robotlar, dronlar, yumuşak robotlar ve `MJCF (.xml)`, `URDF`, `.obj`, `.glb`, `.ply`, `.stl` vb. dosya türlerini yükleme desteği.
- **Foto-gerçekçi render**: Yerel ışın izleme tabanlı render.
- **Farklılaştırılabilirlik**: Genesis tamamen farklılaştırılabilir olacak şekilde tasarlanmıştır. Şu anda, MPM Çözücü ve Araç Çözücümüz farklılaştırılabilirliği desteklemektedir. Gelecek sürümler için farklı çözücülerin geliştirilmesi planlanmıştır (sert ve mafsallı gövde çözücüsünden başlayarak).
- **Fizik tabanlı dokunsal simülasyon**: Farklılaştırılabilir [dokunma sensörü simülasyonu](https://github.com/Genesis-Embodied-AI/DiffTactile) 0.3.0 sürümünde eklenecek.
- **Kullanıcı dostu**: Basit bir kurulum için ve sezgisellik ve API'ler kullanılarak tasarlandı.

## Hızlı Kurulum

Önce [orijinal kurulum yönergelerini](https://pytorch.org/get-started/locally/) takip ederek **PyTorch** yüklemelisiniz.

Ardından, Genesis'i PyPI aracılığıyla yükleyin:
```bash
pip install genesis-world # Python >=3.9 gerektirir;
```

En son Genesis sürümü için repoyu klonlayın ve yerel olarak yükleyin:

```bash
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e .
```

## Docker

Genesis'i Docker'dan kullanmak istiyorsanız, önce Docker image'ini aşağıdaki şekilde derleyebilirsiniz:

```bash
docker build -t genesis -f docker/Dockerfile docker
```

Daha sonra örnekleri docker image içinde çalıştırabilirsiniz (`/workspace/examples` adresine bağlanacaktır.):

```bash
xhost +local:root # Konteynerin görüntüye erişmesine izin ver

docker run --gpus all --rm -it \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v $PWD:/workspace \
Genesis
```

## Dokümantasyon

Ayrıntılı kurulum adımları, öğreticiler ve API referanslarını içeren kapsamlı dokümantasyon [İngilizce](https://genesis-world.readthedocs.io/en/latest/user_guide/index.html) ve [Çince](https://genesis-world.readthedocs.io/zh-cn/latest/user_guide/index.html) dillerinde mevcuttur.

## Genesis'e Katkıda Bulunmak

Genesis projesi açık ve işbirliğine dayalı bir çalışmadır. Topluluktan gelen her türlü katkıyı memnuniyetle karşılıyoruz:

- Yeni özellikler veya hata düzeltmeleri için **Pull Request**.
- GitHub Issues aracılığıyla **Bug Reports**.
- Genesis'in kullanılabilirliğini geliştirmek için **Suggestions**.

Daha fazla ayrıntı için [katkı kılavuzumuza](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/CONTRIBUTING.md) bakın.

## Destek

- GitHub [Issues](https://github.com/Genesis-Embodied-AI/Genesis/issues) üzerinden hata bildirin veya özellik talep edin.
- GitHub [Discussions](https://github.com/Genesis-Embodied-AI/Genesis/discussions) üzerinden tartışmalara katılın veya soru sorun.

## Lisans ve Teşekkür

Genesis kaynak kodu Apache 2.0 altında lisanslanmıştır.

Genesis'in gelişimi aşağıdaki açık kaynaklı projeler sayesinde mümkün olmuştur:

- [Taichi](https://github.com/taichi-dev/taichi): Yüksek performanslı çapraz platform compute backend. Teknik destekleri için Taichi ekibine teşekkür ediyoruz.
- [FluidLab](https://github.com/zhouxian/FluidLab): Referans MPM Solver implementasyonu.
- [SPH_Taichi](https://github.com/erizmr/SPH_Taichi): Referans SPH Solver implementasyonu.
- [Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html) ve [PBF3D](https://github.com/WASD4959/PBF3D): Referans PBD Solver implementasyonu.
- [MuJoCo](https://github.com/google-deepmind/mujoco): Rijit gövde dinamiği için referans.
- [libccd](https://github.com/danfis/libccd): Çarpışma tespiti için referans.
- [PyRender](https://github.com/mmatl/pyrender): Tarama tabanlı işleyici.
- [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) ve [LuisaRender](https://github.com/LuisaGroup/LuisaRender): Işın izleme DSL'i.

## Alakalı Makaleler

Genesis, mevcut ve devam eden çeşitli araştırma çalışmalarının en son teknolojilerini tek bir sisteme entegre eden büyük ölçekli bir projedir. Aşağıda Genesis projesine katkıda bulunan tüm makalelerin genel bir listesini sunuyoruz:

- Xian, Zhou, et al. "Fluidlab: A differentiable environment for benchmarking complex fluid manipulation." arXiv preprint arXiv:2303.02346 (2023).
- Xu, Zhenjia, et al. "Roboninja: Learning an adaptive cutting policy for multi-material objects." arXiv preprint arXiv:2302.11553 (2023).
- Wang, Yufei, et al. "Robogen: Towards unleashing infinite data for automated robot learning via generative simulation." arXiv preprint arXiv:2311.01455 (2023).
- Wang, Tsun-Hsuan, et al. "Softzoo: A soft robot co-design benchmark for locomotion in diverse environments." arXiv preprint arXiv:2303.09555 (2023).
- Wang, Tsun-Hsuan Johnson, et al. "Diffusebot: Breeding soft robots with physics-augmented generative diffusion models." Advances in Neural Information Processing Systems 36 (2023): 44398-44423.
- Katara, Pushkal, Zhou Xian, and Katerina Fragkiadaki. "Gen2sim: Scaling up robot learning in simulation with generative models." 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024.
- Si, Zilin, et al. "DiffTactile: A Physics-based Differentiable Tactile Simulator for Contact-rich Robotic Manipulation." arXiv preprint arXiv:2403.08716 (2024).
- Wang, Yian, et al. "Thin-Shell Object Manipulations With Differentiable Physics Simulations." arXiv preprint arXiv:2404.00451 (2024).
- Lin, Chunru, et al. "UBSoft: A Simulation Platform for Robotic Skill Learning in Unbounded Soft Environments." arXiv preprint arXiv:2411.12711 (2024).
- Zhou, Wenyang, et al. "EMDM: Efficient motion diffusion model for fast and high-quality motion generation." European Conference on Computer Vision. Springer, Cham, 2025.
- Qiao, Yi-Ling, Junbang Liang, Vladlen Koltun, and Ming C. Lin. "Scalable differentiable physics for learning and control." International Conference on Machine Learning. PMLR, 2020.
- Qiao, Yi-Ling, Junbang Liang, Vladlen Koltun, and Ming C. Lin. "Efficient differentiable simulation of articulated bodies." In International Conference on Machine Learning, PMLR, 2021.
- Qiao, Yi-Ling, Junbang Liang, Vladlen Koltun, and Ming Lin. "Differentiable simulation of soft multi-body systems." Advances in Neural Information Processing Systems 34 (2021).
- Wan, Weilin, et al. "Tlcontrol: Trajectory and language control for human motion synthesis." arXiv preprint arXiv:2311.17135 (2023).
- Wang, Yian, et al. "Architect: Generating Vivid and Interactive 3D Scenes with Hierarchical 2D Inpainting." arXiv preprint arXiv:2411.09823 (2024).
- Zheng, Shaokun, et al. "LuisaRender: A high-performance rendering framework with layered and unified interfaces on stream architectures." ACM Transactions on Graphics (TOG) 41.6 (2022): 1-19.
- Fan, Yingruo, et al. "Faceformer: Speech-driven 3d facial animation with transformers." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
- Wu, Sichun, Kazi Injamamul Haque, and Zerrin Yumak. "ProbTalk3D: Non-Deterministic Emotion Controllable Speech-Driven 3D Facial Animation Synthesis Using VQ-VAE." Proceedings of the 17th ACM SIGGRAPH Conference on Motion, Interaction, and Games. 2024.
- Dou, Zhiyang, et al. "C· ase: Learning conditional adversarial skill embeddings for physics-based characters." SIGGRAPH Asia 2023 Conference Papers. 2023.

... ve devam eden birçok çalışma.

## Alıntı

Araştırmanızda Genesis'i kullanıyorsanız, lütfen alıntı yapmayı unutmayın:

```bibtex
@software{Genesis,
  author = {Genesis Authors},
  title = {Genesis: A Universal and Generative Physics Engine for Robotics and Beyond},
  month = {December},
  year = {2024},
  url = {https://github.com/Genesis-Embodied-AI/Genesis}
}
