# Example Running Outputs<img src="./heading-divider.svg" alt="" width="100%" height="1">

## 2.2.11 Experiment 11: (Discussion 5 in Section 5.5) Comparison with Alternative Model Scaling Techniques<img src="./heading-divider.svg" alt="" width="100%" height="1">

### Minimal working example

> **Key observation:** VLASelect achieves higher accuracy than knowledge distillation and dynamic pruning techniques.

<table align="center">
  <tbody>
    <tr>
      <td align="center">Compared knowledge distillation techniques</td>
      <td align="center">Logit distillation; Feature distillation; Attention distillation; Data distillation; MiniLLM; DistiLLM</td>
    </tr>
    <tr>
      <td align="center">Compared dynamic pruning techniques</td>
      <td align="center">LLM in a Flash; PowerInfer; LLM Pruner; EdgeTA</td>
    </tr>
    <tr>
      <td align="center">VLASelect's accuracy improvement than these techniques</td>
      <td align="center"><strong>57.19%</strong></td>
    </tr>
  </tbody>
</table>

<div align="center">
  <img src="../imgs/3.1.2-mwe.png" alt="Scaling strategies on all methods" style="zoom:33%;" />
</div>

<br><br>
