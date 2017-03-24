import pyparsing as pp
from pyparsing import ParseException, pyparsing_common as ppc


class Parser:
    def __init__(self):
        self.grad_points = []
        self.parser = self.build_parser()

    def build_parser(self):
        number = ppc.fraction | ppc.number

        short_hex_color = pp.Suppress('#') + pp.Word(pp.nums + pp.hexnums, exact=3)
        short_hex_color.addParseAction(lambda t: tuple(int(ch+ch, 16) for ch in t[0]))
        long_hex_color = pp.Suppress('#') + pp.Word(pp.nums + pp.hexnums, exact=6)
        long_hex_color.addParseAction(self.long_hex_color)
        hex_color = long_hex_color | short_hex_color

        int_or_percent = (ppc.integer + pp.Literal('%')('percent')) | ppc.integer
        int_or_percent.addParseAction(lambda t: t[0] * 255 / 100 if t.percent else t[0])

        rgb_color_keyword = pp.Suppress('rgba(') | pp.Suppress('rgb(')
        rgb_color = rgb_color_keyword + pp.delimitedList(int_or_percent) + pp.Suppress(')')
        rgb_color.addParseAction(lambda t: (t[0], t[1], t[2]))

        color = hex_color ^ rgb_color

        grad_point = number('x') + pp.Optional(':') + color('y')
        grad_point.addParseAction(lambda t: self.grad_points.append((t.x, t.y)))

        grad_points = pp.OneOrMore(grad_point + pp.lineEnd())
        return grad_points

    def parse(self, text):
        return self.parser.parseString(text, parseAll=True)

    @staticmethod
    def long_hex_color(t):
        s = t[0]
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return r, g, b
