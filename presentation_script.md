# Script thuyết trình project Crypto Price Prediction

## Tổng thời gian: khoảng 10 phút

> Mục tiêu script: giải thích rõ các thuật ngữ, mô hình, đặc trưng dữ liệu, phương pháp đánh giá, tác dụng và lý do sử dụng trong project dự báo giá BTC/ETH.

---

# 1. Giới thiệu tổng quan đề tài

Kính thưa thầy/cô,

Đề tài của em là **dự báo giá Cryptocurrency bằng Machine Learning**, tập trung vào hai đồng chính là **Bitcoin - BTCUSDT** và **Ethereum - ETHUSDT**.

Lý do em chọn đề tài này là vì thị trường crypto có tính biến động cao, dữ liệu thay đổi liên tục và chịu ảnh hưởng bởi nhiều yếu tố như cung cầu, tâm lý thị trường, tin tức, dòng tiền và yếu tố vĩ mô. Vì vậy, đây là một bài toán phù hợp để áp dụng các mô hình dự báo chuỗi thời gian.

Mục tiêu của project không chỉ là dự đoán giá, mà còn là xây dựng một pipeline hoàn chỉnh gồm:

- Thu thập dữ liệu.
- Tạo đặc trưng kỹ thuật.
- Huấn luyện nhiều mô hình khác nhau.
- Đánh giá mô hình bằng nhiều metric.
- Kiểm tra xem mô hình có thật sự tốt hơn phương pháp dự báo đơn giản hay không.

Điểm quan trọng trong project là em không chỉ nhìn vào chỉ số đẹp như R² cao, mà còn dùng **walk-forward validation** và **naive baseline** để đánh giá sát với thực tế tài chính hơn.

---

# 2. Các thuật ngữ và dữ liệu đầu vào

## 2.1. Cryptocurrency

**Cryptocurrency** là tài sản số sử dụng công nghệ blockchain. Trong project này, em chọn Bitcoin và Ethereum vì đây là hai đồng có vốn hóa lớn, thanh khoản cao và có dữ liệu lịch sử đầy đủ.

Tác dụng của việc chọn BTC và ETH là giúp dữ liệu đủ tin cậy để huấn luyện model, đồng thời kết quả có tính đại diện hơn so với các đồng coin nhỏ có thanh khoản thấp.

## 2.2. Time Series

Bài toán này là bài toán **time series forecasting**, tức là dự báo giá trị tương lai dựa trên dữ liệu quá khứ.

Trong time series, thứ tự thời gian rất quan trọng. Không thể random shuffle dữ liệu như các bài toán machine learning thông thường, vì nếu trộn dữ liệu sẽ làm mô hình nhìn thấy thông tin tương lai, gây sai lệch đánh giá.

Vì vậy, project xử lý dữ liệu theo đúng trình tự thời gian: dùng dữ liệu quá khứ để dự đoán dữ liệu tương lai.

## 2.3. OHLCV

Dữ liệu đầu vào chính là dữ liệu **OHLCV** hằng ngày:

- **Open**: giá mở cửa trong ngày.
- **High**: giá cao nhất trong ngày.
- **Low**: giá thấp nhất trong ngày.
- **Close**: giá đóng cửa trong ngày.
- **Volume**: khối lượng giao dịch.

Lý do dùng OHLCV là vì đây là dạng dữ liệu chuẩn trong tài chính, phản ánh đầy đủ biến động giá và mức độ giao dịch trong từng phiên.

Trong project, giá **Close** là biến quan trọng nhất vì thường được dùng làm đại diện cho giá cuối ngày và là cơ sở để tính return.

## 2.4. Return và Log-return

Thay vì dự đoán trực tiếp giá ngày tiếp theo, project dự đoán **return**, tức là mức thay đổi tương đối của giá.

Lý do dùng return là vì giá BTC và ETH có scale lớn, ví dụ BTC có thể ở mức vài chục nghìn USD, còn ETH thấp hơn nhiều. Nếu dự đoán trực tiếp giá, model dễ bị ảnh hưởng bởi scale tuyệt đối.

Return giúp dữ liệu ổn định hơn, dễ so sánh hơn và phù hợp hơn với bài toán tài chính.

Sau khi dự đoán return, em chuyển lại thành giá dự đoán bằng công thức:

```text
predicted_price = base_close * (1 + predicted_return)
```

Tác dụng của cách này là mô hình học biến động tương đối, sau đó vẫn có thể biểu diễn kết quả dưới dạng giá để dễ trực quan hóa.

---

# 3. Các đặc trưng sử dụng trong project

## 3.1. Feature engineering là gì?

**Feature engineering** là quá trình tạo thêm các biến đầu vào từ dữ liệu gốc để giúp model học được nhiều thông tin hơn.

Nếu chỉ dùng giá Close, model chỉ nhìn thấy một chiều thông tin. Khi thêm các chỉ báo kỹ thuật, model có thể học thêm về xu hướng, động lượng, biến động và trạng thái quá mua/quá bán của thị trường.

## 3.2. Moving Average - MA7 và MA14

**Moving Average** là trung bình động của giá trong một khoảng thời gian.

Trong project có:

- **MA7**: trung bình giá 7 ngày.
- **MA14**: trung bình giá 14 ngày.

Tác dụng:

- Làm mượt dữ liệu giá.
- Giúp nhận biết xu hướng ngắn hạn.
- Giảm ảnh hưởng của nhiễu từng ngày.

Lý do dùng MA7 và MA14 là vì crypto biến động nhanh, nên các window ngắn phù hợp để bắt xu hướng gần đây.

## 3.3. RSI

**RSI - Relative Strength Index** là chỉ báo đo sức mạnh tăng/giảm của giá, thường dùng để nhận biết vùng quá mua hoặc quá bán.

Tác dụng:

- RSI cao có thể cho thấy thị trường đang quá mua.
- RSI thấp có thể cho thấy thị trường đang quá bán.
- Giúp model nhận biết trạng thái momentum của thị trường.

Lý do dùng RSI là vì trong thị trường tài chính, hành vi tăng/giảm quá mạnh thường có khả năng điều chỉnh, nên RSI có thể cung cấp thêm tín hiệu cho model.

## 3.4. MACD

**MACD - Moving Average Convergence Divergence** là chỉ báo dùng để nhận biết xu hướng và momentum dựa trên sự khác biệt giữa các đường trung bình động.

Tác dụng:

- Phát hiện xu hướng tăng hoặc giảm.
- Nhận biết thay đổi động lượng của thị trường.
- Hỗ trợ model phân biệt giai đoạn trend mạnh và trend yếu.

Lý do dùng MACD là vì crypto thường có các giai đoạn tăng/giảm mạnh, và MACD giúp biểu diễn thông tin xu hướng tốt hơn giá thô.

## 3.5. Bollinger Bands

**Bollinger Bands** gồm ba đường: dải trên, dải giữa và dải dưới, được tính dựa trên moving average và độ lệch chuẩn.

Tác dụng:

- Đo mức độ biến động của giá.
- Khi dải Bollinger mở rộng, thị trường biến động mạnh.
- Khi dải thu hẹp, thị trường biến động thấp.
- Giá chạm dải trên/dưới có thể gợi ý trạng thái quá mua/quá bán.

Lý do dùng Bollinger Bands là vì volatility là yếu tố rất quan trọng trong crypto. Model cần biết thị trường đang biến động mạnh hay yếu để dự báo tốt hơn.

## 3.6. Volume

**Volume** là khối lượng giao dịch.

Tác dụng:

- Cho biết mức độ quan tâm của thị trường.
- Xu hướng tăng kèm volume cao thường đáng tin hơn xu hướng tăng với volume thấp.
- Volume bất thường có thể báo hiệu sự kiện hoặc biến động mạnh.

Lý do dùng volume là vì giá không nên được phân tích tách rời khỏi khối lượng giao dịch.

---

# 4. Các mô hình được sử dụng

## 4.1. Linear Regression

**Linear Regression** là mô hình hồi quy tuyến tính, giả định đầu ra là tổ hợp tuyến tính của các đặc trưng đầu vào.

Tác dụng trong project:

- Làm baseline machine learning đơn giản.
- Huấn luyện nhanh, dễ giải thích.
- Dùng để so sánh với các mô hình deep learning phức tạp hơn.

Lý do dùng Linear Regression là vì trong nghiên cứu ML, cần có baseline đơn giản. Nếu mô hình phức tạp không vượt qua baseline đơn giản thì chưa thể kết luận mô hình phức tạp có giá trị thực tế.

## 4.2. RNN

**RNN - Recurrent Neural Network** là mạng nơ-ron dùng cho dữ liệu tuần tự. RNN xử lý từng timestep và truyền hidden state từ bước trước sang bước sau.

Tác dụng:

- Học quan hệ giữa các ngày liên tiếp.
- Phù hợp với dữ liệu chuỗi thời gian hơn mô hình tuyến tính thông thường.

Lý do dùng RNN là vì giá crypto có tính phụ thuộc thời gian, giá hôm nay có thể liên quan đến xu hướng các ngày trước.

Hạn chế của RNN là khó học quan hệ dài hạn do vấn đề vanishing gradient.

## 4.3. LSTM

**LSTM - Long Short-Term Memory** là phiên bản cải tiến của RNN, có các cổng như input gate, forget gate và output gate.

Tác dụng:

- Ghi nhớ thông tin quan trọng trong chuỗi dài hơn RNN.
- Loại bỏ bớt thông tin không cần thiết.
- Phù hợp với time series tài chính.

Lý do dùng LSTM là vì dữ liệu giá có thể phụ thuộc không chỉ vào vài ngày gần nhất mà còn vào xu hướng trước đó. LSTM giúp model học được dependency dài hơn.

## 4.4. BiLSTM

**BiLSTM - Bidirectional LSTM** là LSTM hai chiều, học thông tin theo cả chiều tiến và chiều ngược trong sequence.

Tác dụng:

- Khai thác pattern trong toàn bộ cửa sổ 20 ngày.
- Có thể học quan hệ giữa đầu và cuối sequence tốt hơn LSTM một chiều.

Lý do dùng BiLSTM là để kiểm tra xem việc học hai chiều trong window dữ liệu có giúp cải thiện dự báo không.

Trong bài toán dự báo thực tế, ta không dùng dữ liệu tương lai ngoài window. BiLSTM chỉ học hai chiều trong cửa sổ quá khứ đã có, nên không vi phạm nguyên tắc dự báo.

## 4.5. Transformer

**Transformer** là mô hình sử dụng **attention mechanism**. Thay vì xử lý tuần tự như RNN/LSTM, Transformer học mức độ quan trọng giữa các timestep trong sequence.

Tác dụng:

- Tập trung vào những ngày quan trọng trong 20 ngày gần nhất.
- Học quan hệ xa trong chuỗi tốt hơn RNN truyền thống.
- Có khả năng mô hình hóa pattern phức tạp.

Lý do dùng Transformer là vì attention đang được dùng nhiều trong các bài toán sequence, và em muốn so sánh nó với LSTM/BiLSTM trong bài toán dự báo crypto.

## 4.6. Vì sao dùng nhiều mô hình?

Em dùng nhiều mô hình để so sánh theo nhiều mức độ phức tạp:

- Linear Regression: đơn giản, dễ giải thích.
- RNN: sequence model cơ bản.
- LSTM: sequence model có memory tốt hơn.
- BiLSTM: học hai chiều trong window.
- Transformer: attention-based model.

Tác dụng của việc so sánh này là giúp trả lời câu hỏi: **mô hình càng phức tạp có thật sự dự báo tốt hơn không?**

Kết quả cho thấy trong crypto, mô hình phức tạp chưa chắc tốt hơn nếu feature chưa đủ mạnh hoặc dữ liệu có nhiều noise.

---

# 5. Quy trình huấn luyện và xử lý dữ liệu

## 5.1. Sliding Window

Project dùng **sliding window** với độ dài 20 ngày.

Ý nghĩa:

```text
Dùng 20 ngày gần nhất để dự đoán return của ngày tiếp theo.
```

Tác dụng:

- Biến dữ liệu time series thành supervised learning.
- Giữ được thứ tự thời gian.
- Phù hợp với các model sequence như RNN, LSTM, BiLSTM và Transformer.

Lý do chọn window 20 ngày là vì đây xấp xỉ số ngày giao dịch trong một tháng, đủ để model nhìn được xu hướng ngắn hạn mà không làm input quá dài.

## 5.2. Scaling

**Scaling** là chuẩn hóa dữ liệu đầu vào để các feature có cùng thang đo.

Tác dụng:

- Giúp mô hình học ổn định hơn.
- Tránh feature có scale lớn chi phối feature khác.
- Đặc biệt cần thiết cho neural network.

Project dùng scaler và đã sửa lỗi data leakage bằng cách:

- Chia train/test trước.
- Fit scaler chỉ trên tập train.
- Transform tập test bằng scaler đã học từ train.

Lý do làm như vậy là để mô hình không nhìn thấy phân phối dữ liệu tương lai.

## 5.3. Data Leakage

**Data leakage** là hiện tượng thông tin từ tập test hoặc tương lai bị rò rỉ vào quá trình training.

Tác hại:

- Làm kết quả đánh giá cao giả tạo.
- Mô hình có vẻ tốt trên báo cáo nhưng kém khi dùng thực tế.

Trong project, lỗi leakage ban đầu nằm ở việc scaler fit trên toàn bộ dataset. Sau khi sửa, đánh giá trở nên công bằng hơn.

---

# 6. Các phương pháp đánh giá

## 6.1. Train/Test Split

**Train/test split** là chia dữ liệu thành tập huấn luyện và tập kiểm thử.

Tác dụng:

- Train model trên dữ liệu quá khứ.
- Kiểm tra model trên dữ liệu chưa học.

Trong time series, việc chia phải theo thứ tự thời gian, không được shuffle.

## 6.2. Walk-forward Validation

**Walk-forward validation** là phương pháp đánh giá gần với thực tế tài chính hơn.

Cách hoạt động:

- Train trên một đoạn quá khứ.
- Test trên đoạn tương lai gần.
- Dịch cửa sổ thời gian và lặp lại nhiều fold.

Tác dụng:

- Mô phỏng quá trình dự báo thật.
- Kiểm tra model qua nhiều giai đoạn thị trường.
- Giảm rủi ro đánh giá may mắn trên một tập test duy nhất.

Lý do dùng walk-forward là vì thị trường crypto thay đổi theo thời gian, nên một lần train/test split chưa đủ để đánh giá độ ổn định của model.

## 6.3. Naive Baseline

**Naive baseline** là phương pháp dự báo đơn giản, ví dụ lấy giá hoặc return gần nhất làm dự báo cho bước tiếp theo.

Tác dụng:

- Làm mốc so sánh tối thiểu.
- Kiểm tra model ML có thật sự tạo thêm giá trị hay không.

Lý do dùng naive baseline là vì trong tài chính, nếu model phức tạp không thắng được dự báo đơn giản thì model đó chưa có ý nghĩa thực tế.

---

# 7. Các metric đánh giá

## 7.1. MAE

**MAE - Mean Absolute Error** là sai số tuyệt đối trung bình.

Ý nghĩa:

```text
MAE càng thấp thì dự báo càng gần giá trị thực tế.
```

Tác dụng:

- Dễ hiểu vì cùng đơn vị với giá.
- Ví dụ MAE = 1,349 USD nghĩa là trung bình dự báo BTC lệch khoảng 1,349 USD.

Lý do dùng MAE là vì nó trực quan và ít bị ảnh hưởng bởi outlier hơn RMSE.

## 7.2. RMSE

**RMSE - Root Mean Squared Error** là căn bậc hai của sai số bình phương trung bình.

Tác dụng:

- Phạt mạnh các lỗi dự báo lớn.
- Phù hợp khi muốn chú ý đến các lần model dự đoán lệch nhiều.

Lý do dùng RMSE là vì trong tài chính, sai số lớn có thể gây rủi ro cao hơn sai số nhỏ.

## 7.3. R²

**R² - hệ số xác định** đo mức độ mô hình giải thích được biến động của dữ liệu.

Ý nghĩa:

- R² càng gần 1 thì model fit dữ liệu càng tốt.
- R² nhỏ hoặc âm nghĩa là model kém.

Tuy nhiên, với dữ liệu giá tài chính, R² cao chưa chắc model dự báo tốt. Vì giá có autocorrelation cao, giá hôm nay thường gần giá hôm qua nên model có thể đạt R² cao nhưng vẫn không dự đoán tốt biến động thật.

Vì vậy, em không chỉ dùng R² mà còn dùng MASE và walk-forward validation.

## 7.4. MASE

**MASE - Mean Absolute Scaled Error** là metric so sánh lỗi của model với lỗi của naive baseline.

Ý nghĩa:

- MASE < 1: model tốt hơn naive baseline.
- MASE = 1: model tương đương naive baseline.
- MASE > 1: model kém hơn naive baseline.

Tác dụng:

- Đánh giá model có thật sự hữu ích không.
- So sánh được giữa BTC và ETH dù scale giá khác nhau.

Lý do dùng MASE là vì nó phù hợp với time series forecasting và giúp tránh việc chỉ nhìn vào MAE/R².

## 7.5. Directional Accuracy

**Directional Accuracy** đo tỷ lệ mô hình dự đoán đúng chiều tăng hoặc giảm của giá.

Ý nghĩa:

- Nếu giá thực tế tăng và model cũng dự đoán tăng thì đúng.
- Nếu giá thực tế giảm và model cũng dự đoán giảm thì đúng.

Tác dụng:

- Quan trọng với trading, vì đôi khi dự đoán đúng hướng còn quan trọng hơn dự đoán đúng giá tuyệt đối.

Nếu Directional Accuracy khoảng 50%, model gần giống đoán ngẫu nhiên.

## 7.6. Skill Score

**Skill Score** đo mức cải thiện của model so với baseline.

Tác dụng:

- Cho biết model có tạo thêm giá trị so với phương pháp đơn giản không.
- Nếu skill score âm, model đang kém hơn baseline.

Lý do dùng skill score là để đánh giá tính thực dụng của model trong bối cảnh tài chính.

---

# 8. Kết quả và ý nghĩa

Trên tập test thông thường, các mô hình đạt R² khá cao, khoảng 0.94 đến 0.97. Điều này cho thấy model fit khá tốt với dữ liệu test theo cách chia thông thường.

Một số kết quả nổi bật:

- Với BTC, LSTM đạt MAE khoảng 1,349 USD và R² khoảng 0.956.
- Với ETH, Transformer đạt MAE khoảng 53 USD và R² khoảng 0.970.

Tuy nhiên, khi dùng walk-forward validation, kết quả cho thấy các mô hình chưa vượt qua naive baseline:

- MASE phần lớn lớn hơn 1.
- Directional Accuracy xấp xỉ 50%.
- Mô hình deep learning không luôn tốt hơn Linear Regression.

Ý nghĩa của kết quả này là:

- Project đã chứng minh được R² cao chưa đủ để kết luận model tốt trong tài chính.
- Đánh giá bằng walk-forward validation giúp phát hiện overfitting rõ hơn.
- Technical indicators cơ bản chưa đủ mạnh để dự đoán crypto ổn định.
- Cần bổ sung thêm dữ liệu sentiment, on-chain, macro và kiểm tra bằng backtesting thực tế hơn.

Đây là một kết quả có giá trị, vì trong nghiên cứu tài chính, phát hiện mô hình chưa đủ tốt cũng quan trọng như tìm được mô hình tốt. Nó giúp tránh đưa ra kết luận sai dựa trên metric lạc quan.

---

# 9. Tác dụng của hệ thống đã xây dựng

Project đã xây dựng được một hệ thống tương đối đầy đủ gồm:

- Data pipeline lấy dữ liệu từ Binance.
- Feature engineering tạo chỉ báo kỹ thuật.
- Training pipeline cho nhiều model.
- Model registry để lưu model, scaler và metadata.
- Visualization để tạo biểu đồ Actual vs Predicted, MAE, R², MASE, Directional Accuracy.
- API để gọi dự đoán.
- Dashboard để demo trực quan.

Tác dụng của hệ thống là tạo nền tảng hoàn chỉnh cho bài toán dự báo tài chính. Sau này có thể thay feature, thêm model hoặc cải thiện validation mà không cần viết lại toàn bộ project.

---

# 10. Kết luận

Tóm lại, project sử dụng dữ liệu OHLCV của BTC và ETH, tạo các đặc trưng kỹ thuật như MA, RSI, MACD, Bollinger Bands và Volume, sau đó huấn luyện nhiều mô hình từ Linear Regression đến RNN, LSTM, BiLSTM và Transformer.

Các mô hình được đánh giá bằng MAE, RMSE, R², MASE, Directional Accuracy và Skill Score. Trong đó, walk-forward validation và MASE là hai phần quan trọng nhất vì giúp kiểm tra model có thật sự tốt hơn naive baseline hay không.

Kết quả cho thấy model có R² cao trên test split, nhưng chưa vượt qua naive baseline trong walk-forward validation. Vì vậy, kết luận chính là: **mô hình deep learning và technical indicators cơ bản chưa đủ để dự báo giá crypto ổn định trong thực tế**.

Tuy nhiên, project đã đạt được mục tiêu quan trọng là xây dựng pipeline ML hoàn chỉnh, phát hiện data leakage, đánh giá mô hình nghiêm túc và tạo nền tảng để phát triển tiếp.

Em xin hết phần trình bày. Em cảm ơn thầy/cô đã lắng nghe.

---

# Câu hỏi giáo viên có thể hỏi và gợi ý trả lời

## Câu 1: Vì sao dùng return thay vì giá?

Vì return ổn định hơn giá tuyệt đối và giảm ảnh hưởng của scale. Giá BTC và ETH khác nhau rất nhiều, còn return giúp mô hình học biến động tương đối.

## Câu 2: Vì sao cần naive baseline?

Vì naive baseline là mốc tối thiểu. Nếu model phức tạp không thắng được baseline đơn giản thì model chưa có giá trị thực tế.

## Câu 3: Vì sao R² cao nhưng MASE lại kém?

Vì R² trên price space dễ bị ảnh hưởng bởi autocorrelation. Model có thể dự đoán giá gần đúng nhưng không dự đoán tốt biến động tương lai. MASE so trực tiếp với naive baseline nên phản ánh giá trị dự báo thực tế hơn.

## Câu 4: Vì sao dùng walk-forward validation?

Vì time series phải đánh giá theo thứ tự thời gian. Walk-forward mô phỏng cách dự báo thực tế: chỉ dùng quá khứ để dự đoán tương lai.

## Câu 5: Vì sao LSTM/Transformer không chắc tốt hơn Linear Regression?

Vì model phức tạp cần dữ liệu và feature đủ mạnh. Nếu dữ liệu nhiều noise hoặc feature chưa chứa signal tốt, deep learning có thể overfit và không vượt được baseline đơn giản.

## Câu 6: Nếu cải thiện project, em sẽ làm gì?

Em sẽ bổ sung sentiment analysis, on-chain metrics, macro data, dữ liệu intraday và backtesting có transaction cost để đánh giá khả năng ứng dụng thực tế.
