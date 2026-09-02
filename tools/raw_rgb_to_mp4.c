// Minimal RGB24-to-MP4 encoder using the FFmpeg libraries already present on
// the host. It reads width*height*3 byte frames from stdin.
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>

#include <stdio.h>
#include <stdlib.h>

static int write_packet(AVCodecContext *codec_ctx, AVFormatContext *fmt_ctx,
                        AVStream *stream, AVFrame *frame) {
  int ret = avcodec_send_frame(codec_ctx, frame);
  if (ret < 0) return ret;
  AVPacket *packet = av_packet_alloc();
  if (!packet) return AVERROR(ENOMEM);
  while (ret >= 0) {
    ret = avcodec_receive_packet(codec_ctx, packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
    if (ret < 0) { av_packet_free(&packet); return ret; }
    av_packet_rescale_ts(packet, codec_ctx->time_base, stream->time_base);
    packet->stream_index = stream->index;
    ret = av_interleaved_write_frame(fmt_ctx, packet);
    av_packet_unref(packet);
    if (ret < 0) { av_packet_free(&packet); return ret; }
  }
  av_packet_free(&packet);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s output.mp4 width height fps < frames.rgb\n", argv[0]);
    return 2;
  }
  const char *output = argv[1];
  const int width = atoi(argv[2]), height = atoi(argv[3]), fps = atoi(argv[4]);
  if (width <= 0 || height <= 0 || fps <= 0 || width % 2 || height % 2) return 2;

  AVFormatContext *fmt_ctx = NULL;
  AVCodecContext *codec_ctx = NULL;
  AVFrame *frame = NULL;
  struct SwsContext *sws = NULL;
  uint8_t *rgb = NULL;
  int status = 1;

  avformat_alloc_output_context2(&fmt_ctx, NULL, NULL, output);
  if (!fmt_ctx) goto done;
  const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
  if (!codec) goto done;
  AVStream *stream = avformat_new_stream(fmt_ctx, NULL);
  codec_ctx = avcodec_alloc_context3(codec);
  if (!stream || !codec_ctx) goto done;
  codec_ctx->codec_id = AV_CODEC_ID_MPEG4;
  codec_ctx->bit_rate = 4000000;
  codec_ctx->width = width;
  codec_ctx->height = height;
  codec_ctx->time_base = (AVRational){1, fps};
  codec_ctx->framerate = (AVRational){fps, 1};
  codec_ctx->gop_size = 12;
  codec_ctx->max_b_frames = 0;
  codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
    codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  if (avcodec_open2(codec_ctx, codec, NULL) < 0) goto done;
  if (avcodec_parameters_from_context(stream->codecpar, codec_ctx) < 0) goto done;
  stream->time_base = codec_ctx->time_base;
  if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE) && avio_open(&fmt_ctx->pb, output, AVIO_FLAG_WRITE) < 0) goto done;
  if (avformat_write_header(fmt_ctx, NULL) < 0) goto done;

  frame = av_frame_alloc();
  if (!frame) goto done;
  frame->format = codec_ctx->pix_fmt;
  frame->width = width;
  frame->height = height;
  if (av_frame_get_buffer(frame, 32) < 0) goto done;
  sws = sws_getContext(width, height, AV_PIX_FMT_RGB24, width, height,
                       AV_PIX_FMT_YUV420P, SWS_BILINEAR, NULL, NULL, NULL);
  rgb = malloc((size_t)width * height * 3);
  if (!sws || !rgb) goto done;

  for (int64_t frame_index = 0;
       fread(rgb, 1, (size_t)width * height * 3, stdin) == (size_t)width * height * 3;
       ++frame_index) {
    if (av_frame_make_writable(frame) < 0) goto done;
    const uint8_t *src[] = {rgb};
    const int src_stride[] = {width * 3};
    sws_scale(sws, src, src_stride, 0, height, frame->data, frame->linesize);
    frame->pts = frame_index;
    if (write_packet(codec_ctx, fmt_ctx, stream, frame) < 0) goto done;
  }
  if (!feof(stdin)) goto done;
  if (write_packet(codec_ctx, fmt_ctx, stream, NULL) < 0) goto done;
  if (av_write_trailer(fmt_ctx) < 0) goto done;
  status = 0;
done:
  if (status) fprintf(stderr, "raw_rgb_to_mp4: encoding failed\n");
  free(rgb);
  sws_freeContext(sws);
  av_frame_free(&frame);
  avcodec_free_context(&codec_ctx);
  if (fmt_ctx) {
    if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE) && fmt_ctx->pb) avio_closep(&fmt_ctx->pb);
    avformat_free_context(fmt_ctx);
  }
  return status;
}
