TODO:
等测试完以后，修改得到新的代码
仿照之前work/test_half_resolution/main_controller.py的逻辑，加上这个代码里面的回滚逻辑。

代码说明
data_collector.py 连接robot、camera、wowskin_sensor，获得observation和send_action
data_processor.py 拼接数据，以及build_payload_with_two_frames
rollback_manager.py 
- 维护rollback_limited和rollback_happened
- force_ratio_threshold控制实际force / 预测force < hreshold的时候执行rollback
- max_consecutive_failures
- reset_wait_time复位后等待一段时间让机器人稳定

ws_client.py 服务器通讯
main_controller.py
- 保存force的对比图
- 复位：先张开夹爪等待一段时间防止物体抓紧，然后再复位。

