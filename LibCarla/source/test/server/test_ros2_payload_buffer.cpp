// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include <carla/ros2/middleware/fastdds/GenericCdrPubSubType.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

using carla::ros2::GenericCdrPubSubType;
namespace msg = carla::ros2::msg;

template<typename Message, typename Validate>
void CheckPayload(Message message, Validate validate) {
  GenericCdrPubSubType<Message> type;
  const auto size = type.getSerializedSizeProvider(&message)();
  ASSERT_EQ(size, carla::ros2::serialize_to_cdr(message).size());
  // CDR leaves alignment padding untouched. Give both buffers the same
  // padding so the byte comparison only depends on the serialized message.
  std::vector<uint8_t> reference(size, 0xabu);
  eprosima::fastcdr::FastBuffer reference_buffer(
      reinterpret_cast<char*>(reference.data()), reference.size());
  eprosima::fastcdr::Cdr reference_cdr(
      reference_buffer,
      eprosima::fastcdr::Cdr::LITTLE_ENDIANNESS,
      eprosima::fastcdr::CdrVersion::XCDRv1);
  reference_cdr.serialize_encapsulation();
  carla::ros2::serialize_cdr(reference_cdr, message);

  for (const bool with_representation : {false, true}) {
    typename GenericCdrPubSubType<Message>::SerializedPayload_t payload(size + 16u);
    payload.max_size = size;
    std::fill(payload.data, payload.data + size + 16u, 0xabu);
    auto* const buffer = payload.data;
    const bool ok = with_representation
        ? type.serialize(&message, &payload, eprosima::fastdds::dds::XCDR_DATA_REPRESENTATION)
        : type.serialize(&message, &payload);
    ASSERT_TRUE(ok);
    EXPECT_EQ(payload.data, buffer);
    ASSERT_EQ(payload.length, size);
    EXPECT_EQ(payload.encapsulation, CDR_LE);
    EXPECT_TRUE(std::equal(reference.begin(), reference.end(), payload.data));
    EXPECT_EQ(payload.data[0], 0u);
    EXPECT_EQ(payload.data[1], 1u);
    EXPECT_EQ(payload.data[2], 0u);
    EXPECT_EQ(payload.data[3], 0u);
    Message recovered{};
    ASSERT_TRUE(carla::ros2::deserialize_from_cdr(payload.data, payload.length, recovered));
    validate(recovered);
    EXPECT_TRUE(std::all_of(payload.data + size, payload.data + size + 16u,
                           [](uint8_t value) { return value == 0xabu; }));
  }
}

} // namespace

TEST(ros2_payload_buffer, fixed_message_matches_classic_cdr) {
  msg::Clock message{};
  message.clock.sec = 42;
  message.clock.nanosec = 123456789u;
  CheckPayload(message, [&](const msg::Clock& recovered) {
    EXPECT_EQ(recovered.clock.sec, message.clock.sec);
    EXPECT_EQ(recovered.clock.nanosec, message.clock.nanosec);
  });
}

TEST(ros2_payload_buffer, full_hd_image_uses_the_instance_size) {
  msg::Image image{};
  image.header.frame_id = "camera";
  image.height = 1080;
  image.width = 1920;
  image.encoding = "bgra8";
  image.step = image.width * 4;
  image.data.assign(image.step * image.height, 0x5au);
  ASSERT_GT(carla::ros2::cdr_serialized_size(image),
            carla::ros2::CdrTopicInfo<msg::Image>::max_serialized_size());
  CheckPayload(image, [&](const msg::Image& recovered) {
    EXPECT_EQ(recovered.header.frame_id, image.header.frame_id);
    EXPECT_EQ(recovered.height, image.height);
    EXPECT_EQ(recovered.width, image.width);
    EXPECT_EQ(recovered.encoding, image.encoding);
    EXPECT_EQ(recovered.step, image.step);
    EXPECT_EQ(recovered.data, image.data);
  });
}

TEST(ros2_payload_buffer, large_point_cloud_uses_the_instance_size) {
  msg::PointCloud2 cloud{};
  cloud.header.frame_id = "lidar";
  cloud.height = 1;
  cloud.width = 100000;
  cloud.fields = {{"x", 0, msg::PointField::FLOAT32, 1},
                  {"y", 4, msg::PointField::FLOAT32, 1},
                  {"z", 8, msg::PointField::FLOAT32, 1}};
  cloud.point_step = 12;
  cloud.row_step = cloud.width * cloud.point_step;
  cloud.data.assign(cloud.row_step, 0x6bu);
  cloud.is_dense = true;
  ASSERT_GT(carla::ros2::cdr_serialized_size(cloud),
            carla::ros2::CdrTopicInfo<msg::PointCloud2>::max_serialized_size());
  CheckPayload(cloud, [&](const msg::PointCloud2& recovered) {
    EXPECT_EQ(recovered.header.frame_id, cloud.header.frame_id);
    EXPECT_EQ(recovered.height, cloud.height);
    EXPECT_EQ(recovered.width, cloud.width);
    ASSERT_EQ(recovered.fields.size(), cloud.fields.size());
    for (size_t i = 0; i < cloud.fields.size(); ++i) {
      EXPECT_EQ(recovered.fields[i].name, cloud.fields[i].name);
      EXPECT_EQ(recovered.fields[i].offset, cloud.fields[i].offset);
      EXPECT_EQ(recovered.fields[i].datatype, cloud.fields[i].datatype);
      EXPECT_EQ(recovered.fields[i].count, cloud.fields[i].count);
    }
    EXPECT_EQ(recovered.point_step, cloud.point_step);
    EXPECT_EQ(recovered.row_step, cloud.row_step);
    EXPECT_EQ(recovered.is_dense, cloud.is_dense);
    EXPECT_EQ(recovered.data, cloud.data);
  });
}

TEST(ros2_payload_buffer, insufficient_capacity_does_not_overrun) {
  msg::Clock message{};
  GenericCdrPubSubType<msg::Clock> type;
  const auto size = type.getSerializedSizeProvider(&message)();
  for (const uint32_t capacity : {0u, 3u, size - 1u}) {
    GenericCdrPubSubType<msg::Clock>::SerializedPayload_t payload(size + 16u);
    std::fill(payload.data, payload.data + size + 16u, 0xabu);
    payload.max_size = capacity;
    EXPECT_FALSE(type.serialize(&message, &payload));
    EXPECT_TRUE(std::all_of(payload.data + capacity, payload.data + size + 16u,
                           [](uint8_t value) { return value == 0xabu; }));
  }
}

TEST(ros2_payload_buffer, absent_payload_storage_is_rejected) {
  msg::Clock message{};
  GenericCdrPubSubType<msg::Clock> type;
  GenericCdrPubSubType<msg::Clock>::SerializedPayload_t payload;
  ASSERT_EQ(payload.data, nullptr);
  EXPECT_FALSE(type.serialize(&message, &payload));
}
