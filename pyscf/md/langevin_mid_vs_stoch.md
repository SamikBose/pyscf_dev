Format is Stochastic Dynamics equation first then steps to rearrange to Langevin Middle.

---

$$
m_{i} \frac{d^{2}\mathbf{r}_{i}}{dt^{2}}=-m_{i}\gamma_{i} \frac{d\mathbf{r}_{i}}{dt}+\mathbf{F}_{i}(\mathbf{r})+\dot{\mathbf{r}}_{i}
$$

$$
\therefore\quad m_i\frac{d\mathbf{v}_i}{dt}=\mathbf{f}_i-\gamma m_i \mathbf{v}_i+\mathbf{R}_i
$$

Same as Langevin Middle.

---

$$
\dot{\mathbf{r}}_{i}(t): \left< \dot{r}_{i}(t)\dot{r}_{j}(t+s) \right> =2m_{i}\gamma_{i}k_{B}T \delta(s)\delta_{ij}
$$

In Langevin Middle:

$$
\mathbf{R}_{i}=\mathcal{N}(\mu=0, \sigma=2 \gamma k_{B}T)
$$

Stochastic Dynamics multiplies by the mass here. Also what is $\delta(s)$ and $\delta_{ij}$? I am assuming $s$ is the step size. Something strange is that $\dot{\mathbf{r}}_{i}$ never seems to be used in the equations below. A different $\mathbf{r}$ is used instead.

---

$$
(1) \quad \mathbf{v}'=\mathbf{v}\left( t-\frac{1}{2}\Delta t \right)+\frac{1}{m}\mathbf{F}(t)\Delta t
$$

$$
(2) \quad \Delta \mathbf{v}=-\alpha \mathbf{v}'\left( t+\frac{1}{2}\Delta t \right)+\left( \sqrt{ \frac{k_{B}T}{m}\alpha(2-\alpha) } \right)\mathbf{r}^{G}_{i}
$$

$$
(3) \quad \mathbf{v}\left( t+\frac{1}{2}\Delta t \right)=\mathbf{v}'+\Delta \mathbf{v}
$$

$$
\alpha=1-e^{-\gamma \Delta t} \qquad \mathbf{r}^{G}_{i}=\mathcal{N}(\mu=0,\sigma=1)
$$

Langevin Middle does:

$$
(1) \quad \mathbf{v}_{i}(t+\Delta t/2) = \mathbf{v}_{i}(t-\Delta t/2) + \mathbf{f}_{i}(t)\Delta t/{m}_{i}
$$

$$
(2) \quad \mathbf{v'}_{i}(t+\Delta t/2) = \mathbf{v}_{i}(t+\Delta t/2)\alpha + \left(\sqrt{kT(1-\alpha^2)/m}\right)R
$$

$$
\alpha=e^{-\gamma \Delta t} \qquad \mathbf{R}_{i}=\mathcal{N}(\mu=0, \sigma=2 \gamma k_{B}T)
$$

Noticably $\alpha$ and $\mathbf{R}_{i}$ are different between the two. The first equation looks the same to find $\mathbf{v}'$ assuming that $\mathbf{v}'$ in this case is refering to $\mathbf{v}'(t+\Delta t/2)$. The second equation looks similar except the first term is negative and the last term with the square root looks quite a bit different. Langevin Middle also seems to be missing this $\hspace{0pt}3$rd equation which calculates the final next half step velocity.

Decided to ask ChatGPT and Claude if they were equal and they both said yes.

- https://chatgpt.com/share/6a0df0e6-2fc0-83ea-a4e1-e46f49c712e2
- https://claude.ai/share/725aa494-196a-43df-b5ba-51fd32d6057b
