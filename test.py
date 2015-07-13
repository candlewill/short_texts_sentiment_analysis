import qrcode
from qrcode.image.pure import PymagingImage
import scipy


def to_qrcode(texts):
    qr = qrcode.QRCode(
        version=7,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=0,
    )
    if len(texts) <= 140:
        qr.add_data(texts)
        qr.make(fit=False)
        img = qr.make_image()
        return img
    else:
        print('Texts too long :' + texts)
        return


if __name__ == '__main__':
    img = to_qrcode('Hello')
    a = list(img.getdata())
    print(len(a), a)
    with open('./data/pic/test.png', 'wb') as f:
        img.save(f)
