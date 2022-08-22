import numpy
import PIL

def draw_boxes(img, boxes, digits, digit_width, digit_height):
    img_out = img.copy()
    draw = PIL.ImageDraw.Draw(img_out)
    for i in range(len(boxes)):
        color = numpy.random.randint(0, 255, 3)
        x = boxes[i]
        draw.polygon([
                (x-digit_width/2, 3), 
                (x+digit_width/2, 3), 
                (x+digit_width/2, digit_height-3), 
                (x-digit_width/2, digit_height-3)
            ], 
            outline=(color[0], color[1], color[2], 0))
        draw.text((x-digit_width/2+2, 5), str(digits[i]), fill=(0,0,0,128))
    return img_out

def decode_output(confidence, box_shift, digit, position_width, positions, digit_width):
    thr = 0.5
    
    ret_dig = []
    ret_box = []
    
    ditits_bb = (box_shift-0.5)*2*position_width + (0.5+numpy.arange(positions))*position_width
    
    while 1:
        max_confidence = numpy.argmax(confidence)
        if(confidence[max_confidence] < thr):
            break
        max_pos = ditits_bb[max_confidence]

        ret_dig.append(digit[max_confidence])
        ret_box.append(ditits_bb[max_confidence])

        merge_box = numpy.where(numpy.abs(ditits_bb-max_pos) < digit_width/2)
        confidence[merge_box] = 0
    if(len(ret_box)):
        ret_box, ret_dig = zip(*sorted(zip(ret_box, ret_dig)))
    
    return (ret_dig, ret_box)
