# Example Running Outputs<img src="./heading-divider.svg" alt="" width="100%" height="1">

## 2.2.15 Extra Experiment 1: Impact of Transferred Updates on the Large Model<img src="./heading-divider.svg" alt="" width="100%" height="1">

### Minimal working example

> **Key observation:** The transferred update tends to improve the large model's accuracy. <br>This is due to VLASelect's selective knowledge transfer mechanism based on neuron indexes.

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

|  | large model's accuracy before feedback | large model's accuracy after feedback |
| :---: | :---: | :---: |
| Feedback 1 | 0.8333 | 0.8333 |
| Feedback 2 | 0.3333 | 0.6667 |
