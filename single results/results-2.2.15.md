# Example Running Outputs<img src="./heading-divider.svg" alt="" width="100%" height="1">

## 2.2.15 Extra Experiment 1: Impact of Transferred Updates on the Large Model<img src="./heading-divider.svg" alt="" width="100%" height="1">

### Minimal working example

> **Key observation:** The transferred update can improve the large model's accuracy.

<!-- <table align="center">
  <tbody>
    <tr>
      <td align="center">VLASelect's accuracy improvement on MLP than ConRFT</td>
      <td align="center"><strong>33.62%</strong></td>
    </tr>
    <tr>
      <td align="center">VLASelect's accuracy improvement on CNN than ConRFT</td>
      <td align="center"><strong>34.72%</strong></td>
    </tr>
  </tbody>
</table> -->

<!-- <div align="center">
  <img src="../imgs/2.2.14.png" alt="" style="zoom:33%;" />
</div> -->

Outputs are shown in the terminal:
```bash
[impact] update=3 large_model_accuracy_before_feedback=0.8333 large_model_accuracy_after_feedback=0.8333 improvement=+0.0000
[impact] update=5 large_model_accuracy_before_feedback=0.3333 large_model_accuracy_after_feedback=0.6667 improvement=+0.3333
```